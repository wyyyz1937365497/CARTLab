/* ============================================================
 * CARTLab — 主控（main.js）
 * 启动流程、事件接线、运行 / 对比 / 导出处理、就绪指示。
 *
 * 就绪选择器（供外部验证）：
 *   html[data-ready="1"]          UI 就绪（ECharts 已加载、事件已接线）
 *   html[data-api="1"]            pywebview 后端已连接
 *   #card-tree[data-result="1"]   已有训练结果渲染
 *   #status-pill[data-state]      boot | ok | busy | error
 *   window.__CART__.uiReady / apiReady / hasResult / frameIndex
 *   window.__CART__.selectNode(id) / runCompare()
 * ============================================================ */
(function (global) {
  'use strict';

  var $ = global.UI.$;
  var $all = global.UI.$all;
  var el = global.UI.el;
  var fmtNum = global.UI.fmtNum;
  var fmtPct = global.UI.fmtPct;

  /** 选中的节点：先查最终模型，再查剪枝前的完整建树（标记已剪掉）。 */
  function lookupSelectedNode() {
    var id = global.State.selectedNodeId;
    if (id === null || id === undefined) return null;
    var n = global.State.findNode(id);
    if (n) return { node: n, prunedAway: false };
    var bn = global.State.findBuildNode(id);
    if (bn) return { node: bn, prunedAway: true };
    return null;
  }

  /** 当前轨道的帧列表（建树帧或剪枝序列）。 */
  function trackFrames(result) {
    return global.State.track === 'prune' ? buildPruneFrames(result) : result.frames;
  }

  /* ---------------- 建树 / 剪枝双轨道 ---------------- */

  /** 把 prune.steps 映射为时间轴伪帧（每个 α 一帧，携带该步树快照）。 */
  function buildPruneFrames(result) {
    var prune = result && result.prune;
    var steps = prune && Array.isArray(prune.steps) ? prune.steps : [];
    var chosen = prune && prune.chosen_alpha != null ? prune.chosen_alpha : null;
    var frames = [];
    steps.forEach(function (s) {
      if (!s || !s.tree || typeof s.tree !== 'object') return;
      frames.push({
        event: 'prune',
        _track: 'prune',
        alpha: s.alpha,
        leaves: s.leaves != null ? s.leaves : null,
        depth: s.depth,
        nodes: s.nodes,
        chosen: s.chosen === true ||
          (chosen != null && s.alpha != null && Math.abs(s.alpha - chosen) < 1e-12),
        val_accuracy: s.val_accuracy,
        train_accuracy: s.train_accuracy,
        pruned_node: s.pruned_node != null ? s.pruned_node : null,
        tree: s.tree
      });
    });
    return frames;
  }

  /** 剪枝轨道可用：CCP 启用且存在带树快照的剪枝步。 */
  function pruneTrackAvailable() {
    var r = global.State.result;
    if (!r || !r.prune || !r.prune.enabled || !Array.isArray(r.prune.steps) || !r.prune.steps.length) {
      return false;
    }
    return buildPruneFrames(r).length > 0;
  }

  /** 同步轨道按钮的可见性与选中态。 */
  function updateTrackUI() {
    var wrap = $('#tl-track');
    if (!wrap) return;
    wrap.hidden = !pruneTrackAvailable();
    $('#tl-track-build').classList.toggle('active', global.State.track !== 'prune');
    $('#tl-track-prune').classList.toggle('active', global.State.track === 'prune');
  }

  /** 切换轨道：重载帧序列；剪枝轨道落到选中 α 的那一步，建树轨道落最后。 */
  function setTrack(track) {
    if (!global.State.result) return;
    if (track !== 'build' && !pruneTrackAvailable()) track = 'build';
    if (track === global.State.track) { updateTrackUI(); return; }
    global.State.track = track;
    updateTrackUI();
    var frames = trackFrames(global.State.result);
    global.Timeline.load(frames);
    global.Timeline.go(trackLandingIndex(frames));
  }

  /** 轨道初始落点：剪枝轨道 = chosen 的那一步（兜底最后一步），建树轨道 = 最后。 */
  function trackLandingIndex(frames) {
    if (global.State.track !== 'prune') return frames.length - 1;
    for (var i = 0; i < frames.length; i++) {
      if (frames[i].chosen) return i;
    }
    return frames.length - 1;
  }

  /* ---------------- 帧渲染总入口 ---------------- */

  function renderAll(payload) {
    var frame = payload.frame, index = payload.index, total = payload.total;
    global.State.frameIndex = index;

    if (!frame) {
      $('#tl-pos').textContent = '0 / 0';
      var chip0 = $('#tl-event');
      chip0.textContent = '—';
      chip0.className = 'badge ev';
      $('#tree-note').textContent = '—';
      return;
    }

    var isPrune = frame._track === 'prune';
    var hit = lookupSelectedNode();
    global.Charts.renderTree(frame, global.State.selectedNodeId);
    global.Panels.renderNodeDetail(
      hit ? Object.assign({}, hit.node, hit.prunedAway ? { pruned_away: true } : null) : null,
      global.State.datasetInfo);
    global.Charts.renderCandidates(hit && !hit.prunedAway ? hit.node : null);

    $('#tl-pos').textContent = index + ' / ' + Math.max(total - 1, 0);
    var chip = $('#tl-event');
    if (isPrune) {
      chip.textContent = 'α=' + global.UI.fmtAlpha(frame.alpha) + (frame.chosen ? '*' : '') +
        ' · ' + (frame.leaves != null ? frame.leaves : '—') + ' 叶';
      chip.className = 'badge ev ev-prune';
      $('#tree-note').textContent = (total ? '帧 ' + (index + 1) + '/' + total : '') +
        ' · 剩 ' + (frame.leaves != null ? frame.leaves : '—') + ' 叶' +
        (frame.chosen ? ' · 选中 α*' : '');
    } else {
      chip.textContent = global.UI.eventLabel(frame.event);
      chip.className = 'badge ev ev-' + String(frame.event || 'none');
      $('#tree-note').textContent = (total ? '帧 ' + (index + 1) + '/' + total : '') +
        (frame.node != null ? ' · 节点 #' + frame.node : '');
    }
  }

  /** 只重绘树与节点面板（选中节点变化时）。 */
  function redrawSelection() {
    var cur = global.Timeline.current();
    if (!cur.frame || !global.State.result) return;
    global.Charts.renderTree(cur.frame, global.State.selectedNodeId);
    var hit = lookupSelectedNode();
    global.Panels.renderNodeDetail(
      hit ? Object.assign({}, hit.node, hit.prunedAway ? { pruned_away: true } : null) : null,
      global.State.datasetInfo);
    global.Charts.renderCandidates(hit && !hit.prunedAway ? hit.node : null);
  }

  /** 选中节点（供树点击与 __CART__.selectNode 使用）。 */
  function selectNode(nodeId) {
    if (!global.State.result) return;
    global.State.selectedNodeId = String(nodeId);
    redrawSelection();
  }

  /* ---------------- 评估面板（分类指标 + 混淆矩阵） ---------------- */

  function refreshEvalPanels() {
    if (!global.State.result) return;
    var splitKey = $('#sel-eval-split').value || 'test';
    var report = global.State.result.metrics && global.State.result.metrics[splitKey];
    global.Panels.renderClassMetrics(global.State.result, splitKey, global.State.datasetInfo);
    global.Charts.renderConfusion(report);
  }

  /* ---------------- 推理路径 ---------------- */

  function refreshPaths() {
    if (!global.State.result) return;
    var exps = global.State.result.explanation || [];
    var ex = exps[global.State.sampleIndex] || null;
    global.Panels.renderSampleList(global.State.result, global.State.datasetInfo, global.State.sampleIndex);
    global.Panels.renderPathSteps(ex, global.State.datasetInfo);
  }

  /* ---------------- 数据集 ---------------- */

  async function initDatasets() {
    var list = await global.Bridge.call('list_datasets');
    var ids = Array.isArray(list) ? list : [];
    if (!ids.length) {
      global.UI.toastWarn('后端未返回任何数据集');
      return;
    }
    var first = typeof ids[0] === 'string' ? ids[0] : (ids[0] && ids[0].id);
    await loadDatasetInfo(first || 'iris');
  }

  async function loadDatasetInfo(id) {
    global.State.datasetId = id;
    try {
      var info = await global.Bridge.getDatasetInfo(id);
      global.State.datasetInfo = info;
      global.Panels.renderDatasetSummary(info);
    } catch (e) {
      global.UI.toastError('数据集信息加载失败：' + e.message);
    }
  }

  /* ---------------- 配置收集 ---------------- */

  function intVal(sel, dft) {
    var raw = $(sel).value;
    if (raw === null || raw === undefined || String(raw).trim() === '') return dft;
    var v = parseInt(raw, 10);
    return isNaN(v) ? dft : v;
  }

  function floatOrNull(sel) {
    var raw = $(sel).value;
    if (raw === null || raw === undefined || String(raw).trim() === '') return null;
    var v = parseFloat(raw);
    return isNaN(v) ? null : v;
  }

  function collectConfig() {
    var maxDepth = floatOrNull('#inp-max-depth');
    if (maxDepth !== null) {
      maxDepth = Math.round(maxDepth);
      if (maxDepth < 0 || maxDepth > 12) throw new Error('最大深度必须为 0–12 的整数或留空');
    }
    var minLeaf = intVal('#inp-min-leaf', 1);
    if (!(minLeaf >= 1) || minLeaf > 50) throw new Error('叶节点最小样本数必须为 1–50');
    var minGain = floatOrNull('#inp-min-gain');
    if (minGain === null) minGain = 0;
    if (!isFinite(minGain) || minGain < 0) throw new Error('min_gain 必须为非负数');
    var seed = intVal('#inp-seed', 42);
    if (seed < 0) throw new Error('随机种子必须 ≥ 0');
    var alpha = floatOrNull('#inp-alpha');
    if (alpha !== null && (!isFinite(alpha) || alpha < 0)) throw new Error('α 必须为非负数');
    return {
      max_depth: maxDepth,
      min_samples_leaf: minLeaf,
      min_gain: minGain,
      seed: seed,
      ccp: $('#chk-ccp').checked,
      alpha: alpha
    };
  }

  function validateResult(res) {
    if (!res || typeof res !== 'object') throw new Error('后端返回数据不完整');
    if (!res.tree || typeof res.tree !== 'object') throw new Error('后端返回数据不完整：缺少 tree');
    if (!Array.isArray(res.nodes) || !res.nodes.length) throw new Error('后端返回数据不完整：缺少 nodes');
    if (!Array.isArray(res.frames) || !res.frames.length) throw new Error('后端返回数据不完整：缺少 frames');
    if (!res.metrics || !res.metrics.test) throw new Error('后端返回数据不完整：缺少 metrics');
    if (!res.pca || !Array.isArray(res.pca.points) || !res.pca.points.length) {
      throw new Error('后端返回数据不完整：缺少 pca.points');
    }
  }

  /* ---------------- 运行训练 ---------------- */

  async function runTraining() {
    if (global.State.running) return;
    var cfg;
    try { cfg = collectConfig(); }
    catch (e) { global.UI.toastWarn(e.message); return; }

    var btn = $('#btn-run');
    global.State.running = true;
    global.UI.setLoading(btn, true, '⏳ 训练中…');
    global.UI.setStatus('busy', '训练中…');
    try {
      var res = await global.Bridge.call('run_training', cfg);
      validateResult(res);
      global.State.applyResult(res);
      $('#card-tree').dataset.result = '1';
      $('#btn-export-tree').disabled = false;
      $('#btn-export-json').disabled = false;

      global.Panels.renderMetrics(res, global.State.datasetInfo);
      refreshEvalPanels();
      global.Charts.renderPca(res.pca);
      var evr = (res.pca && res.pca.explained_variance_ratio) || [];
      $('#pca-note').textContent = evr.length >= 2
        ? 'PC1 ' + fmtPct(evr[0]) + ' · PC2 ' + fmtPct(evr[1])
        : '—';
      global.State.sampleIndex = 0;
      refreshPaths();

      // 时间轴：按可用轨道载入帧序列（CCP 启用时默认剪枝轨道），
      // 剪枝轨道落到选中 α 的那一步，建树轨道落到最终帧
      global.State.track = pruneTrackAvailable() ? 'prune' : 'build';
      updateTrackUI();
      var frames = trackFrames(res);
      global.Timeline.load(frames);
      // 自动选中根节点：节点详情与候选划分立即有内容
      selectNode(res.tree.id != null ? res.tree.id : 0);
      global.Timeline.go(trackLandingIndex(frames));

      var st = res.stats || {};
      global.UI.toastSuccess('训练完成：' + st.nodes + ' 节点 / ' + st.leaves + ' 叶 · 深度 ' + st.depth +
        ' · 验证准确率 ' + fmtPct(res.metrics.val && res.metrics.val.accuracy) +
        ' · ' + fmtNum(st.build_ms, 1) + ' ms');
      global.UI.setStatus('ok', '就绪');
    } catch (e) {
      global.UI.toastError(e.message);
      global.UI.setStatus('error', '运行失败');
    } finally {
      global.UI.setLoading(btn, false);
      global.State.running = false;
    }
  }

  /* ---------------- 对比实验 ---------------- */

  async function runCompare() {
    if (global.State.comparing) return;
    if (global.State.running) { global.UI.toastWarn('训练正在运行，请稍候'); return; }
    var cfg;
    try { cfg = collectConfig(); }
    catch (e) { global.UI.toastWarn(e.message); return; }

    var btn = $('#btn-compare');
    global.State.comparing = true;
    global.UI.setLoading(btn, true, '⏳ 对比实验进行中…');
    global.UI.setStatus('busy', '对比实验中…');
    try {
      var cmp = await global.Bridge.call('compare_variants', cfg);
      if (!cmp || !Array.isArray(cmp.variants) || !cmp.variants.length) {
        throw new Error('对比实验返回数据不完整');
      }
      global.State.compare = cmp;
      $('#card-compare').dataset.open = '1';
      global.Panels.renderCompare(cmp);
      if (global.Charts.ensureCompareCharts()) {
        global.Charts.renderPruneCurve(cmp);
      }
      rafResize();
      $('#card-compare').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      global.UI.toastSuccess('对比完成：' + cmp.variants.length + ' 种变体 · α* = ' +
        global.UI.fmtAlpha(cmp.chosen_alpha));
      global.UI.setStatus('ok', '就绪');
    } catch (e) {
      global.UI.toastError(e.message);
      global.UI.setStatus('error', '对比失败');
    } finally {
      global.UI.setLoading(btn, false);
      global.State.comparing = false;
    }
  }

  /* ---------------- 导出 ---------------- */

  function fallbackDownload(href, filename) {
    var a = el('a');
    a.href = href;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  async function exportTreePNG() {
    var url = global.Charts.dataURL('tree');
    if (!url) { global.UI.toastWarn('决策树尚未渲染，无法导出'); return; }
    if (global.Bridge.ready()) {
      try {
        var r = await global.Bridge.call('save_chart', url);
        if (r && r.saved) global.UI.toastSuccess('PNG 已保存：' + (r.path || ''));
        else global.UI.toastWarn('PNG 保存未成功' + (r && r.path ? '：' + r.path : ''));
      } catch (e) {
        global.UI.toastError('保存 PNG 失败：' + e.message);
      }
      return;
    }
    fallbackDownload(url, 'cart_tree.png');
  }

  async function exportJSON() {
    if (!global.State.result) { global.UI.toastWarn('尚无训练结果可导出'); return; }
    if (global.Bridge.ready()) {
      try {
        var r = await global.Bridge.call('save_result', global.State.result);
        if (r && r.saved) global.UI.toastSuccess('JSON 已保存：' + (r.path || ''));
        else global.UI.toastWarn('JSON 保存未成功' + (r && r.path ? '：' + r.path : ''));
      } catch (e) {
        global.UI.toastError('保存 JSON 失败：' + e.message);
      }
      return;
    }
    var blob = new Blob([JSON.stringify(global.State.result, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    fallbackDownload(url, 'cart_result.json');
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }

  /* ---------------- 明暗主题 ---------------- */

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('cart-theme', theme); } catch (e) { /* 忽略存储失败 */ }
    var btn = $('#btn-theme');
    if (btn) btn.textContent = theme === 'light' ? '◐ 暗色' : '◐ 亮色';
    if (!global.Charts || !global.Charts.refreshTheme) return;
    global.Charts.refreshTheme();
    // 数据集芯片 / 推理路径的颜色随主题重绘
    global.Panels.renderDatasetSummary(global.State.datasetInfo);
    if (global.State.result) {
      refreshPaths();
      var cur = global.Timeline.current();
      if (cur && cur.frame) renderAll(cur);
      refreshEvalPanels();
      global.Charts.renderPca(global.State.result.pca);
    }
    if (global.State.compare) {
      global.Panels.renderCompare(global.State.compare);
      if (global.Charts.ensureCompareCharts()) {
        global.Charts.renderPruneCurve(global.State.compare);
      }
    }
  }

  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem('cart-theme'); } catch (e) { /* 忽略 */ }
    applyTheme(saved === 'light' ? 'light' : 'dark');
  }

  /* ---------------- 视图切换 ---------------- */

  function switchView(name) {
    document.querySelectorAll('.view-tab').forEach(function (tab) {
      tab.classList.toggle('active', tab.dataset.view === name);
    });
    $('#view-lab').classList.toggle('hidden', name !== 'lab');
    $('#view-explain').classList.toggle('hidden', name !== 'explain');
    if (name === 'lab' && global.Charts.resizeAll) {
      setTimeout(function () { global.Charts.resizeAll(); }, 60);
    }
    window.scrollTo(0, 0);
  }

  /* ---------------- 事件接线 ---------------- */

  function wireDom() {
    $('#btn-run').addEventListener('click', runTraining);
    $('#btn-compare').addEventListener('click', runCompare);
    document.querySelectorAll('.view-tab').forEach(function (tab) {
      tab.addEventListener('click', function () { switchView(tab.dataset.view); });
    });
    $('#btn-theme').addEventListener('click', function () {
      applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
    });
    $('#btn-export-tree').addEventListener('click', exportTreePNG);
    $('#btn-export-json').addEventListener('click', exportJSON);

    // 评估集切换：分类指标表 + 混淆矩阵
    $('#sel-eval-split').addEventListener('change', refreshEvalPanels);

    // 轨道切换：建树回放 / 剪枝序列
    $('#tl-track-build').addEventListener('click', function () { setTrack('build'); });
    $('#tl-track-prune').addEventListener('click', function () { setTrack('prune'); });

    // 时间轴
    $('#tl-first').addEventListener('click', function () { global.Timeline.pause(); global.Timeline.first(); });
    $('#tl-prev').addEventListener('click', function () { global.Timeline.pause(); global.Timeline.prev(); });
    $('#tl-play').addEventListener('click', function () { global.Timeline.toggle(); });
    $('#tl-next').addEventListener('click', function () { global.Timeline.pause(); global.Timeline.next(); });
    $('#tl-last').addEventListener('click', function () { global.Timeline.pause(); global.Timeline.last(); });
    $('#tl-speed').addEventListener('change', function (e) {
      global.Timeline.setSpeed(parseFloat(e.target.value));
    });
    $('#tl-slider').addEventListener('input', function (e) {
      global.Timeline.pause();
      global.Timeline.go(parseInt(e.target.value, 10) || 0);
    });

    // 树节点点击 → 选中
    global.Charts.bindTreeClick(function (nodeId) {
      selectNode(String(nodeId));
    });

    // 推理路径样本选择
    global.Bus.on('selectSample', function (i) {
      global.State.sampleIndex = i;
      refreshPaths();
    });

    // Bus
    Bus.on('frame', renderAll);
    Bus.on('slider', function (p) {
      var s = $('#tl-slider');
      s.max = String(Math.max(p.total - 1, 0));
      s.value = String(p.index);
      s.disabled = p.total === 0;
      ['tl-first', 'tl-prev', 'tl-play', 'tl-next', 'tl-last'].forEach(function (id) {
        $('#' + id).disabled = p.total === 0;
      });
    });
    Bus.on('playstate', function (playing) {
      $('#tl-play').textContent = playing ? '⏸' : '▶';
      $('#tl-play').classList.toggle('active', !!playing);
    });
    Bus.on('toast', function (p) { global.UI.toast(p.kind, p.message); });

    // 键盘：←/→ 步进，空格播放（输入控件内不劫持）
    document.addEventListener('keydown', function (e) {
      var tag = e.target && e.target.tagName;
      if (tag === 'INPUT' && e.target.type !== 'range') return;
      if (tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'BUTTON') return;
      if (!global.State.result) return;
      if (e.key === 'ArrowLeft') { global.Timeline.pause(); global.Timeline.prev(); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { global.Timeline.pause(); global.Timeline.next(); e.preventDefault(); }
      else if (e.key === ' ') { global.Timeline.toggle(); e.preventDefault(); }
    });

    // 窗口尺寸变化
    var resizeTimer = null;
    global.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { global.Charts.resizeAll(); }, 150);
    });
  }

  function rafResize() {
    requestAnimationFrame(function () { global.Charts.resizeAll(); });
  }

  /* ---------------- 启动 ---------------- */

  async function boot() {
    // 1) 本地 ECharts（index.html 已带静态 script 标签；此处兜底探测备用路径）
    var chartsOk = true;
    try {
      await global.Charts.loadECharts();
      global.Charts.initCharts();
    } catch (e) {
      chartsOk = false;
      global.UI.fatal('本地 ECharts 加载失败，图表功能不可用', e.message);
      global.UI.setStatus('error', '图表库缺失');
    }

    // 2) 初始空态面板
    initTheme();
    global.Panels.renderDatasetSummary(null);
    global.Panels.renderMetrics(null, null);
    global.Panels.renderClassMetrics(null, 'test', null);
    global.Panels.renderNodeDetail(null, null);
    global.Panels.renderSampleList(null, null, 0);
    global.Panels.renderPathSteps(null, null);

    // 3) 事件接线
    wireDom();

    // UI 自身就绪（无论后端是否可用），暴露验证选择器
    document.documentElement.dataset.ready = '1';

    // 4) 后端
    if (chartsOk) global.UI.setStatus('boot', '连接后端…');
    try {
      await global.Bridge.wait();
      document.documentElement.dataset.api = '1';
      if (chartsOk) global.UI.setStatus('ok', '后端已连接');
      await initDatasets();
      if (chartsOk) global.UI.setStatus('ok', '就绪');
    } catch (e) {
      document.documentElement.dataset.api = '0';
      global.UI.setStatus('error', '后端不可用');
      global.UI.fatal('无法连接后端（pywebview API）', e.message);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { boot(); }, { once: true });
  } else {
    boot();
  }

  /* ---------------- 验证辅助 ---------------- */

  Object.defineProperty(global, '__CART__', {
    value: {
      get uiReady() { return document.documentElement.dataset.ready === '1'; },
      get apiReady() { return document.documentElement.dataset.api === '1'; },
      get hasResult() { return !!global.State.result; },
      get frameIndex() { return global.State.frameIndex; },
      get track() { return global.State.track; },
      get playing() { return global.Timeline.isPlaying(); },
      get dataset() { return global.State.datasetId; },
      get selectedNode() { return global.State.selectedNodeId; },
      get compareReady() { return !!global.State.compare; },
      selectNode: function (id) { selectNode(String(id)); },
      runCompare: function () { return runCompare(); }
    },
    writable: false,
    enumerable: true,
    configurable: false
  });
})(window);
