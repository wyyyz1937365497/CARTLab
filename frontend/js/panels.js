/* ============================================================
 * CARTLab — 面板渲染（panels.js）
 * DOM 渲染函数（不直接读取输入控件）：
 *  - renderDatasetSummary  数据集摘要 + 特征 / 类别芯片
 *  - renderMetrics         整体运行指标（准确率 / 树规模 / 耗时 / 剪枝）
 *  - renderClassMetrics    逐类别 precision / recall / F1 / support 表
 *  - renderNodeDetail      选中节点详情（划分 / 基尼 / 增益 / 计数）
 *  - renderSampleList      推理路径样本按钮
 *  - renderPathSteps       单样本逐节点判定步骤
 *  - renderCompare         对比实验：变体指标表
 * ============================================================ */
(function (global) {
  'use strict';

  var UI = global.UI;
  var el = UI.el;
  var fmtNum = UI.fmtNum;
  var fmtPct = UI.fmtPct;
  var fmtAlpha = UI.fmtAlpha;

  function addKv(root, k, v, cls) {
    var row = el('div', 'kv-row');
    row.appendChild(el('dt', null, k));
    var dd = el('dd', cls || null, v);
    row.appendChild(dd);
    root.appendChild(row);
    return dd;
  }

  function clearNode(node) { node.textContent = ''; }

  function clsLabel(info, c) {
    if (info && Array.isArray(info.class_labels) && info.class_labels[c]) return info.class_labels[c];
    if (info && Array.isArray(info.classes) && info.classes[c]) return info.classes[c];
    return '类别 ' + c;
  }

  function featLabel(info, f) {
    if (info && Array.isArray(info.feature_labels) && info.feature_labels[f]) return info.feature_labels[f];
    if (info && Array.isArray(info.features) && info.features[f]) return info.features[f];
    return '特征 ' + f;
  }

  /** 类别计数的紧凑形式：山鸢尾 30 · 变色鸢尾 0 … */
  function countsText(counts, info) {
    return (counts || []).map(function (v, i) {
      return clsLabel(info, i) + ' ' + v;
    }).join(' · ');
  }

  /* ---------------- 数据集摘要 ---------------- */

  function renderDatasetSummary(info) {
    global.Charts.setDatasetInfo(info);
    if (!info) {
      ['#ds-name', '#ds-samples', '#ds-features', '#ds-classes'].forEach(function (sel) {
        global.UI.$(sel).textContent = '—';
      });
      global.UI.$('#ds-source').textContent = '—';
      global.UI.$('#ds-source').removeAttribute('href');
      global.UI.$('#ds-description').textContent = '—';
      global.UI.$('#ds-count-badge').textContent = '—';
      clearNode(global.UI.$('#ds-feature-chips'));
      clearNode(global.UI.$('#ds-class-chips'));
      return;
    }
    global.UI.$('#ds-name').textContent = info.name || info.id || '—';
    global.UI.$('#ds-count-badge').textContent = (info.samples != null ? info.samples : '—') + ' 样本';
    global.UI.$('#ds-samples').textContent = fmtNum(info.samples, 0);
    global.UI.$('#ds-features').textContent = fmtNum(info.features ? info.features.length : 0, 0);
    global.UI.$('#ds-classes').textContent = fmtNum(info.classes ? info.classes.length : 0, 0);
    var link = global.UI.$('#ds-source');
    link.textContent = (info.source || '—').replace(/^https?:\/\//, '');
    if (info.source) { link.setAttribute('href', info.source); } else { link.removeAttribute('href'); }
    global.UI.$('#ds-description').textContent = info.description || '—';

    var feats = global.UI.$('#ds-feature-chips');
    clearNode(feats);
    (info.features || []).forEach(function (f, i) {
      var label = (info.feature_labels && info.feature_labels[i]) || f;
      feats.appendChild(el('span', 'chip', label + ' ' + f));
    });

    var cls = global.UI.$('#ds-class-chips');
    clearNode(cls);
    (info.classes || []).forEach(function (c, i) {
      var chip = el('span', 'chip class-chip');
      var dot = el('span', 'dot');
      dot.style.background = global.classColor(i);
      chip.appendChild(dot);
      chip.appendChild(el('span', null,
        ((info.class_labels && info.class_labels[i]) || c) + ' × ' +
        ((info.counts && info.counts[i]) != null ? info.counts[i] : '—')));
      cls.appendChild(chip);
    });
  }

  /* ---------------- 运行指标 ---------------- */

  function renderMetrics(result, info) {
    var kv = global.UI.$('#metrics-kv');
    var hint = global.UI.$('#metrics-hint');
    var badge = global.UI.$('#metrics-alpha');
    clearNode(kv);
    if (!result || !result.metrics) {
      if (hint) hint.classList.remove('hidden');
      badge.textContent = '—';
      return;
    }
    if (hint) hint.classList.add('hidden');
    var m = result.metrics;
    var st = result.stats || {};
    var prune = result.prune || {};

    addKv(kv, '训练准确率', fmtPct(m.train && m.train.accuracy));
    addKv(kv, '验证准确率', fmtPct(m.val && m.val.accuracy));
    addKv(kv, '测试准确率', fmtPct(m.test && m.test.accuracy),
      (m.test && m.test.accuracy) >= 0.9 ? 'good' : null);
    addKv(kv, '宏平均 P / R / F1（测试）', fmtPct(m.test && m.test.macro && m.test.macro.precision) +
      ' / ' + fmtPct(m.test && m.test.macro && m.test.macro.recall) +
      ' / ' + fmtPct(m.test && m.test.macro && m.test.macro.f1));
    addKv(kv, '节点 / 叶 / 深度', (st.nodes != null ? st.nodes : '—') +
      ' / ' + (st.leaves != null ? st.leaves : '—') +
      ' / ' + (st.depth != null ? st.depth : '—'));
    addKv(kv, '评估候选阈值', fmtNum(st.evaluated_thresholds, 0));
    addKv(kv, '构建 / 预测耗时',
      (st.build_ms != null ? fmtNum(st.build_ms, 1) + ' ms' : '—') + ' / ' +
      (st.predict_ms != null ? fmtNum(st.predict_ms, 1) + ' ms' : '—'));
    if (prune.enabled) {
      addKv(kv, '剪枝 α*', fmtAlpha(prune.chosen_alpha));
      if (st.baseline_nodes != null) {
        addKv(kv, '基线 → 剪枝后', st.baseline_nodes + ' 节点 / ' + st.baseline_leaves + ' 叶 → ' +
          st.nodes + ' 节点 / ' + st.leaves + ' 叶');
      }
    } else {
      addKv(kv, '剪枝', '未启用');
    }
    badge.textContent = prune.enabled ? 'α*=' + fmtAlpha(prune.chosen_alpha) : '未剪枝';
  }

  /* ---------------- 分类指标表 ---------------- */

  /** splitKey：train / val / test。 */
  function renderClassMetrics(result, splitKey, info) {
    var tbl = global.UI.$('#tbl-classmetrics');
    clearNode(tbl);
    var report = result && result.metrics && result.metrics[splitKey];
    if (!report) {
      tbl.appendChild(el('caption', 'muted small', '运行训练后显示逐类别指标'));
      return;
    }

    var thead = el('thead');
    var hr = el('tr');
    ['类别', '精确率', '召回率', 'F1', '支持'].forEach(function (h) {
      hr.appendChild(el('th', null, h));
    });
    thead.appendChild(hr);
    tbl.appendChild(thead);

    var tbody = el('tbody');
    function row(name, c, isMacro) {
      var tr = el('tr');
      tr.appendChild(el('td', null, name));
      tr.appendChild(el('td', null, fmtPct(c.precision)));
      tr.appendChild(el('td', null, fmtPct(c.recall)));
      tr.appendChild(el('td', null, fmtPct(c.f1)));
      tr.appendChild(el('td', null, c.support != null ? c.support : '—'));
      tbody.appendChild(tr);
    }
    (report.classes || []).forEach(function (c, i) {
      row(clsLabel(info, i), c);
    });
    if (report.macro) {
      row('宏平均', { precision: report.macro.precision, recall: report.macro.recall, f1: report.macro.f1, support: report.total });
    }
    tbl.appendChild(tbody);
  }

  /* ---------------- 节点详情 ---------------- */

  function renderNodeDetail(node, info) {
    var kv = global.UI.$('#node-detail');
    var hint = global.UI.$('#node-hint');
    clearNode(kv);
    if (!node) {
      if (hint) hint.classList.remove('hidden');
      return;
    }
    if (hint) hint.classList.add('hidden');

    addKv(kv, '节点', '#' + node.id + (node.depth != null ? '（深度 ' + node.depth + '）' : ''));
    if (node.pruned_away) {
      addKv(kv, '剪枝状态', '已剪掉（不在最终模型中）', 'bad');
    }
    addKv(kv, '样本数', node.n != null ? node.n : '—');
    addKv(kv, '类别计数', countsText(node.counts, info));
    addKv(kv, '基尼指数', fmtNum(node.gini, 4));
    if (node.leaf || (node.feature == null && node.threshold == null)) {
      addKv(kv, '预测类别', clsLabel(info, node.prediction), 'good');
      addKv(kv, '停止原因', UI.reasonLabel(node.reason));
    } else {
      addKv(kv, '选中划分', featLabel(info, node.feature) + ' ≤ ' + fmtNum(node.threshold, 4));
      addKv(kv, '增益', fmtNum(node.gain, 4), node.gain > 0 ? 'good' : null);
    }
  }

  /* ---------------- 推理路径 ---------------- */

  function renderSampleList(result, info, selectedIndex) {
    var wrap = global.UI.$('#sample-list');
    var hint = global.UI.$('#paths-hint');
    clearNode(wrap);
    var exps = result && Array.isArray(result.explanation) ? result.explanation : [];
    if (!exps.length) {
      if (hint) hint.classList.remove('hidden');
      return;
    }
    if (hint) hint.classList.add('hidden');
    exps.forEach(function (e, i) {
      var ok = e.true_label === e.predicted;
      var chip = el('button', 'sample-chip' + (i === selectedIndex ? ' active' : ''));
      chip.type = 'button';
      chip.title = '样本 #' + e.index + ' · 真实 ' + clsLabel(info, e.true_label) +
        ' · 预测 ' + clsLabel(info, e.predicted);
      chip.appendChild(el('span', null, '#' + e.index));
      chip.appendChild(el('span', ok ? 'mark-ok' : 'mark-bad', ok ? '✓' : '✘'));
      chip.addEventListener('click', function () {
        global.Bus.emit('selectSample', i);
      });
      wrap.appendChild(chip);
    });
  }

  /** ex：explanation[i]；info：datasetInfo。 */
  function renderPathSteps(ex, info) {
    var box = global.UI.$('#path-steps');
    clearNode(box);
    if (!ex || !Array.isArray(ex.steps) || !ex.steps.length) return;

    var ok = ex.true_label === ex.predicted;
    var head = el('div', 'path-sample');
    head.appendChild(el('span', null, '样本 #' + ex.index));
    head.appendChild(el('span', 'muted small mono', '特征 [' + (ex.row || []).map(function (v) { return fmtNum(v, 1); }).join(', ') + ']'));
    head.appendChild(el('span', null, '真实 ' + clsLabel(info, ex.true_label) + ' · 预测 ' + clsLabel(info, ex.predicted)));
    head.appendChild(el('span', ok ? 'verdict-ok' : 'verdict-bad', ok ? '✓ 判对' : '✘ 判错'));
    box.appendChild(head);

    var steps = ex.steps;
    steps.forEach(function (s, i) {
      var isFinal = i === steps.length - 1;
      var step = el('div', 'path-step ' + (isFinal ? 'final' : 'cond') + (!ok && isFinal ? ' miss' : ''));
      if (!isFinal && s.feature_label) {
        var cond = el('div', 'path-cond');
        cond.appendChild(el('span', null, '节点 #' + s.node + ' · ' + s.feature_label +
          ' = ' + fmtNum(s.value, 2) + ' ≤ ' + fmtNum(s.threshold, 2) + ' '));
        cond.appendChild(el('span', 'go', s.go_left ? '→ 走左子树' : '→ 走右子树'));
        step.appendChild(cond);
        var meta = 'gini ' + fmtNum(s.gini, 3) + ' · n ' + s.n +
          ' · 计数 ' + (s.counts || []).join('/') +
          (s.gain != null ? ' · 增益 ' + fmtNum(s.gain, 4) : '');
        step.appendChild(el('div', 'path-meta', meta));
      } else if (!isFinal) {
        step.appendChild(el('div', 'path-cond', '节点 #' + s.node));
        step.appendChild(el('div', 'path-meta', 'gini ' + fmtNum(s.gini, 3) + ' · n ' + s.n));
      } else {
        var pred = el('div', 'path-pred',
          '叶节点 → 预测 ' + clsLabel(info, s.prediction != null ? s.prediction : ex.predicted));
        step.appendChild(pred);
        step.appendChild(el('div', 'path-meta',
          'gini ' + fmtNum(s.gini, 3) + ' · n ' + s.n +
          ' · 计数 ' + (s.counts || []).join('/') +
          (s.reason ? ' · ' + UI.reasonLabel(s.reason) : '')));
      }
      box.appendChild(step);
    });
  }

  /* ---------------- 对比实验 ---------------- */

  function paramsText(params) {
    if (!params) return '—';
    var parts = [];
    if ('max_depth' in params) {
      parts.push(params.max_depth == null ? '不限深度' : '深度 ' + params.max_depth);
    }
    if ('min_samples_leaf' in params) parts.push('叶 ≥ ' + params.min_samples_leaf);
    if ('alpha' in params && params.alpha != null) parts.push('α* = ' + fmtAlpha(params.alpha));
    return parts.length ? parts.join(' · ') : '—';
  }

  function renderCompare(cmp) {
    var meta = global.UI.$('#compare-meta');
    var tbl = global.UI.$('#tbl-variants');
    clearNode(tbl);
    if (!cmp || !Array.isArray(cmp.variants) || !cmp.variants.length) {
      tbl.appendChild(el('caption', 'muted small', '无对比数据'));
      meta.textContent = '';
      return;
    }
    var sizes = cmp.split_sizes || [];
    meta.textContent = '数据集：' + (cmp.dataset || '—') +
      ' · 划分 ' + sizes.join(' / ') +
      (cmp.chosen_alpha != null ? ' · 选中的 α* = ' + fmtAlpha(cmp.chosen_alpha) : '');

    var rows = cmp.variants;
    var bestVal = null;
    rows.forEach(function (r) {
      if (typeof r.val === 'number' && (bestVal === null || r.val > bestVal)) bestVal = r.val;
    });

    var thead = el('thead');
    var hr = el('tr');
    ['变体', '参数', '深度', '叶节点', '节点数', '训练', '验证', '测试', '耗时 (ms)']
      .forEach(function (h) { hr.appendChild(el('th', null, h)); });
    thead.appendChild(hr);
    tbl.appendChild(thead);

    var tbody = el('tbody');
    rows.forEach(function (r) {
      var tr = el('tr');
      tr.appendChild(el('td', null, r.name || '—'));
      tr.appendChild(el('td', null, paramsText(r.params)));
      tr.appendChild(el('td', null, r.depth != null ? r.depth : '—'));
      tr.appendChild(el('td', null, r.leaves != null ? r.leaves : '—'));
      tr.appendChild(el('td', null, r.nodes != null ? r.nodes : '—'));
      tr.appendChild(el('td', null, fmtPct(r.train)));
      tr.appendChild(el('td', r.val === bestVal ? 'best' : null, fmtPct(r.val)));
      tr.appendChild(el('td', null, fmtPct(r.test)));
      tr.appendChild(el('td', null, r.ms != null ? fmtNum(r.ms, 1) : '—'));
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
  }

  global.Panels = {
    renderDatasetSummary: renderDatasetSummary,
    renderMetrics: renderMetrics,
    renderClassMetrics: renderClassMetrics,
    renderNodeDetail: renderNodeDetail,
    renderSampleList: renderSampleList,
    renderPathSteps: renderPathSteps,
    renderCompare: renderCompare
  };
})(window);
