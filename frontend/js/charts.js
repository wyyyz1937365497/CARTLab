/* ============================================================
 * CARTLab — 图表渲染模块（charts.js）
 * 全部基于本地 ECharts。渲染函数：
 *  - renderTree         决策树（帧快照，pending 节点标「构建中」，点击选中）
 *  - renderCandidates   选中节点的候选划分（加权基尼横条，选中高亮）
 *  - renderConfusion    混淆矩阵热力图（对角线强调）
 *  - renderPca          PCA 散点（测试样本圆环，误分类 × 标记）
 *  - renderPruneCurve   剪枝曲线（准确率双线 + 叶节点柱，双 y 轴）
 * 另提供 initCharts / ensureCompareCharts / dataURL / resizeAll /
 * clearMain / loadECharts（本地脚本探测加载）。
 * ============================================================ */
(function (global) {
  'use strict';

  var UI = global.UI;
  var fmtNum = UI.fmtNum;
  var MONO = "'JetBrains Mono', 'Cascadia Code', 'Fira Code', Consolas, 'Courier New', monospace";

  var C = {
    text: '#f2f2f2',
    textDim: '#a8a8a8',
    textFaint: '#757575',
    axis: '#333333',
    splitLine: '#242424',
    accent: '#f2f2f2',
    bar: '#9a9a9a',
    green: '#8fbf7f',
    amber: '#d8b26a',
    red: '#d97a7a',
    tooltipBg: '#1f1f1f'
  };

  function cssVar(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  /** 从 CSS 变量重读图表配色；切换明暗主题后由 main.js 调用。 */
  function refreshTheme() {
    C.text = cssVar('--text', '#f2f2f2');
    C.textDim = cssVar('--text-dim', '#a8a8a8');
    C.textFaint = cssVar('--text-faint', '#757575');
    C.axis = cssVar('--border', '#333333');
    C.splitLine = cssVar('--border-soft', '#242424');
    C.accent = C.text;
    C.bar = cssVar('--accent-2', '#9a9a9a');
    C.green = cssVar('--green', '#8fbf7f');
    C.amber = cssVar('--amber', '#d8b26a');
    C.red = cssVar('--red', '#d97a7a');
    C.tooltipBg = cssVar('--card-2', '#1f1f1f');
  }

  function isLight() {
    return document.documentElement.dataset.theme === 'light';
  }

  refreshTheme();

  var ech = null;
  var instances = {};
  var OVERLAYS = {
    tree: 'tree-empty',
    candidates: 'candidates-empty',
    confusion: 'confusion-empty',
    pca: 'pca-empty'
  };

  /** 最近一次数据集信息（feature/class 中文标签）。由 main.js 注入。 */
  var DS = null;
  function setDatasetInfo(info) { DS = info || null; }
  function featLabel(f) {
    if (DS && Array.isArray(DS.feature_labels) && DS.feature_labels[f]) return DS.feature_labels[f];
    if (DS && Array.isArray(DS.features) && DS.features[f]) return DS.features[f];
    return '特征 ' + f;
  }
  function clsLabel(c) {
    if (DS && Array.isArray(DS.class_labels) && DS.class_labels[c]) return DS.class_labels[c];
    if (DS && Array.isArray(DS.classes) && DS.classes[c]) return DS.classes[c];
    return '类别 ' + c;
  }

  var SHORT_REASONS = {
    pure: '纯节点',
    max_depth: '深度上限',
    min_samples_split: '样本不足',
    no_valid_split: '无划分',
    min_gain: '增益不足',
    pruned: '已剪枝'
  };

  /* ---------------- 本地 ECharts 探测加载 ---------------- */

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = function () { resolve(src); };
      s.onerror = function () { reject(new Error('加载失败：' + src)); };
      document.head.appendChild(s);
    });
  }

  /** 依次探测本地 echarts 脚本路径，成功即止。 */
  async function loadECharts() {
    if (global.echarts) return;
    var candidates = [
      'vendor/echarts.min.js',
      'vendor/echarts.js',
      'vendor/js/echarts.min.js',
      '../vendor/echarts.min.js',
      '../vendor/echarts.js'
    ];
    var errs = [];
    for (var i = 0; i < candidates.length; i++) {
      try {
        await loadScript(candidates[i]);
        if (global.echarts) return;
        errs.push(candidates[i] + '（无 window.echarts）');
      } catch (e) {
        errs.push(e.message);
      }
    }
    throw new Error('无法加载本地 ECharts，尝试过：' + errs.join('；'));
  }

  /* ---------------- 初始化 ---------------- */

  function makeChart(key, sel) {
    if (instances[key]) return instances[key];
    var dom = document.querySelector(sel);
    if (!dom) return null;
    if (!ech) {
      if (!global.echarts) return null;
      ech = global.echarts;
    }
    instances[key] = ech.init(dom, null, { renderer: 'canvas' });
    return instances[key];
  }

  /** 主视图四图（树 / 候选 / 混淆 / PCA）在启动时初始化。 */
  function initCharts() {
    if (!global.echarts) throw new Error('window.echarts 不存在：请在 frontend/vendor/ 下放置本地 echarts.min.js');
    ech = global.echarts;
    makeChart('tree', '#chart-tree');
    makeChart('candidates', '#chart-candidates');
    makeChart('confusion', '#chart-confusion');
    makeChart('pca', '#chart-pca');
    return instances;
  }

  /** 对比区图表按需初始化（区块展开后才有尺寸）。 */
  function ensureCompareCharts() {
    if (!global.echarts) return false;
    if (!ech) ech = global.echarts;
    makeChart('prune', '#chart-prune');
    return true;
  }

  /* ---------------- 通用小件 ---------------- */

  function setOverlay(name, show) {
    var id = OVERLAYS[name];
    var node = id && document.getElementById(id);
    if (node) node.classList.toggle('show', !!show);
  }

  function tooltipBase() {
    return {
      backgroundColor: C.tooltipBg,
      borderColor: C.axis,
      textStyle: { color: C.text, fontSize: 11 }
    };
  }

  function axisLabelStyle() { return { color: C.textDim, fontSize: 10 }; }

  /** 混淆矩阵刻度：小字号 + 窄宽度自动断行（右栏窄卡片，类名最长 6 字）。 */
  function confTick() {
    return { color: C.textDim, fontSize: 9.5, width: 32, overflow: 'break', lineHeight: 11 };
  }

  function hexToRgba(hex, alpha) {
    var h = String(hex).replace('#', '');
    if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16);
    if (isNaN(n)) return hex;
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + alpha + ')';
  }

  function valueAxis(name) {
    return {
      type: 'value',
      scale: true,
      name: name,
      nameLocation: 'middle',
      nameGap: 22,
      nameTextStyle: { color: C.textFaint, fontSize: 10 },
      axisLabel: axisLabelStyle(),
      axisLine: { lineStyle: { color: C.axis } },
      splitLine: { lineStyle: { color: C.splitLine } }
    };
  }

  /* ---------------- 决策树 ---------------- */

  /**
   * 节点标签：全部文字放在节点框上方的一个多行标签里（position top）。
   * 行1：入边条件（根节点为「根节点」）；行2：gini · n；行3：[类别计数]；
   * 叶节点追加「→ 预测类 · 停止原因」。
   */
  function treeLabel(data) {
    var node = data._raw || {};
    var edge = data._edge || null;
    var lines = [];
    if (edge) {
      lines.push('{c|' + featLabel(edge.feature) + ' ' + edge.op + ' ' + fmtNum(edge.threshold, 2) + '}');
    } else {
      lines.push('{r|根节点}');
    }
    lines.push('{m|gini ' + fmtNum(node.gini, 3) + ' · n=' + node.n + '}');
    if (node.counts) lines.push('{q|[' + node.counts.join('/') + ']}');
    if (node.pending) {
      lines.push('{b|构建中…}');
    } else if (node.feature == null || node.threshold == null) {
      var cls = clsLabel(node.prediction);
      var pred = '{p|→ ' + cls + '}';
      // 类名较长时原因不放同一行（避免标签过宽、右侧裁切），tooltip 中仍有
      if (node.reason && SHORT_REASONS[node.reason] && cls.length <= 5) {
        pred += '{w| · ' + SHORT_REASONS[node.reason] + '}';
      }
      lines.push(pred);
    }
    return lines.join('\n');
  }

  /**
   * edge：{op:'≤'|'>', feature, threshold} —— 父节点划分在本节点入边上
   * 的条件；根节点传 null。
   */
  function treeConvert(node, selectedId, edge) {
    var isSel = selectedId !== null && selectedId !== undefined &&
      String(selectedId) === String(node.id);
    var pending = !!node.pending;
    var internal = !pending && node.feature != null && node.threshold != null;
    var out = {
      name: '#' + node.id,
      id: node.id,
      _raw: node,
      _edge: edge || null,
      itemStyle: {
        color: global.classColor(node.prediction),
        borderColor: isSel ? C.accent : (pending ? C.textFaint : C.axis),
        borderWidth: isSel ? 2 : 1,
        borderType: pending ? 'dashed' : 'solid'
      },
      label: {
        position: 'top',
        distance: 6,
        fontWeight: isSel ? 'bold' : 'normal',
        color: isSel ? C.text : C.textDim
      },
      children: []
    };
    var th = node.threshold, feat = node.feature;
    (node.children || []).forEach(function (ch, i) {
      var cond = (internal && th != null && feat != null)
        ? { op: i === 0 ? '≤' : '>', feature: feat, threshold: th }
        : null;
      out.children.push(treeConvert(ch, selectedId, cond));
    });
    return out;
  }

  /** 绘制树快照（建树帧或剪枝步）；selectedId 高亮，pending 节点虚线 + 「构建中」。 */
  function renderTree(frame, selectedId) {
    var inst = instances.tree;
    if (!inst) return;
    var tree = frame && frame.tree;
    if (!tree || typeof tree !== 'object') {
      inst.clear();
      setOverlay('tree', true);
      return;
    }
    setOverlay('tree', false);
    inst.setOption({
      backgroundColor: 'transparent',
      animationDuration: 240,
      tooltip: Object.assign(tooltipBase(), {
        formatter: function (params) {
          var n = params.data._raw || {};
          var row = '节点 #' + n.id + (n.depth != null ? ' · 深度 ' + n.depth : '');
          var lines = [row,
            '样本 n=' + n.n + ' · gini=' + fmtNum(n.gini, 3),
            '预测：' + clsLabel(n.prediction)];
          if (n.feature != null && n.threshold != null) {
            lines.push('划分：' + featLabel(n.feature) + ' ≤ ' + fmtNum(n.threshold, 2) +
              (n.gain != null ? ' · 增益 ' + fmtNum(n.gain, 4) : ''));
          }
          if (n.pending) lines.push('构建中…');
          else if (n.reason && global.UI.REASON_LABELS[n.reason]) lines.push(global.UI.REASON_LABELS[n.reason]);
          return lines.join('<br/>');
        }
      }),
      series: [{
        type: 'tree',
        data: [treeConvert(tree, selectedId, null)],
        orient: 'LR',
        left: '16%',     // 根节点上方标签（最宽 ~86px）完整落在画布内
        right: '14%',
        top: 56,          // 容纳节点上方最多 4 行标签（4×11 + 距离 6）
        bottom: '6%',
        symbol: 'rect',
        symbolSize: [15, 11],
        initialTreeDepth: -1,
        expandAndCollapse: false,
        lineStyle: { color: C.axis, width: 1.2 },
        label: {
          position: 'top',
          distance: 6,
          color: C.textDim,
          fontSize: 9.5,
          formatter: function (params) { return treeLabel(params.data); },
          rich: {
            c: { fontSize: 9.5, color: C.textDim, lineHeight: 11 },
            r: { fontSize: 9.5, color: C.textFaint, lineHeight: 11 },
            m: { fontSize: 9.5, color: C.textDim, lineHeight: 11 },
            q: { fontSize: 9.5, color: C.textFaint, fontFamily: MONO, lineHeight: 11 },
            p: { fontSize: 9.5, color: C.text, fontWeight: 600, lineHeight: 11 },
            w: { fontSize: 9.5, color: C.textFaint, lineHeight: 11 },
            b: { fontSize: 9.5, color: C.amber, lineHeight: 11 }
          }
        },
        leaves: {
          label: {
            position: 'top',
            distance: 6,
            color: C.textDim,
            fontSize: 9.5,
            formatter: function (params) { return treeLabel(params.data); }
          }
        }
      }]
    }, { notMerge: true });
  }

  /** 树节点点击：回调节点 id。 */
  function bindTreeClick(handler) {
    var inst = instances.tree;
    if (!inst) return;
    inst.off('click');
    inst.on('click', function (params) {
      if (params.data && params.data.id !== undefined) handler(params.data.id);
    });
  }

  /* ---------------- 候选划分（横向条形） ---------------- */

  function candLabel(c) {
    var name = (c.feature != null) ? featLabel(c.feature) : (c.feature_name || '特征 ?');
    return name + ' ≤ ' + fmtNum(c.threshold, 2);
  }

  function sameSplit(a, b) {
    return a && b && a.feature === b.feature && Math.abs(a.threshold - b.threshold) < 1e-9;
  }

  /** node：扁平节点（含 candidates）；无候选（叶）时显示空态。 */
  function renderCandidates(node) {
    var inst = instances.candidates;
    if (!inst) return;
    var cands = node && Array.isArray(node.candidates) ? node.candidates : [];
    if (!cands.length) {
      inst.clear();
      setOverlay('candidates', true);
      return;
    }
    setOverlay('candidates', false);

    var rows = cands.slice().sort(function (a, b) { return a.weighted_gini - b.weighted_gini; });
    var labels = rows.map(candLabel);
    var chosenIdx = -1;
    rows.forEach(function (c, i) {
      if (sameSplit(c, node)) chosenIdx = i;
    });

    inst.setOption({
      backgroundColor: 'transparent',
      animationDuration: 240,
      grid: { left: 8, right: 52, top: 8, bottom: 4, containLabel: true },
      tooltip: Object.assign(tooltipBase(), {
        trigger: 'item',
        formatter: function (params) {
          var c = rows[params.dataIndex] || {};
          var lines = ['<b>' + candLabel(c) + '</b>' +
            (params.dataIndex === chosenIdx ? '　<span style="color:' + C.green + '">✔ 选中划分</span>' : ''),
            '加权基尼：' + fmtNum(c.weighted_gini, 4),
            '增益：' + fmtNum(c.gain, 4),
            '左 ' + c.n_left + ' · 右 ' + c.n_right];
          return lines.join('<br/>');
        }
      }),
      xAxis: {
        type: 'value',
        axisLabel: axisLabelStyle(),
        axisLine: { show: false },
        splitLine: { lineStyle: { color: C.splitLine } }
      },
      yAxis: {
        type: 'category',
        data: labels,
        inverse: true,
        axisLabel: { color: C.textDim, fontSize: 10, fontFamily: MONO },
        axisLine: { lineStyle: { color: C.axis } },
        axisTick: { show: false }
      },
      series: [{
        type: 'bar',
        barMaxWidth: 13,
        data: rows.map(function (c, i) {
          return {
            value: Number(c.weighted_gini.toFixed(6)),
            itemStyle: {
              color: i === chosenIdx ? C.green : hexToRgba(C.bar, 0.5),
              borderRadius: 0
            }
          };
        }),
        label: {
          show: true,
          position: 'right',
          color: C.textFaint,
          fontSize: 9.5,
          fontFamily: MONO,
          formatter: function (p) { return fmtNum(p.value, 4); }
        }
      }]
    }, { notMerge: true });
  }

  /* ---------------- 混淆矩阵热力图 ---------------- */

  /** report：metrics[split]，含 matrix 与 classes。 */
  function renderConfusion(report) {
    var inst = instances.confusion;
    if (!inst) return;
    var matrix = report && report.matrix;
    if (!Array.isArray(matrix) || !matrix.length) {
      inst.clear();
      setOverlay('confusion', true);
      return;
    }
    setOverlay('confusion', false);

    var labels = matrix.map(function (_, i) { return clsLabel(i); });
    var maxV = 0;
    matrix.forEach(function (row) {
      row.forEach(function (v) { if (v > maxV) maxV = v; });
    });
    maxV = maxV || 1;

    var data = [];
    matrix.forEach(function (row, i) {
      row.forEach(function (v, j) {
        var diag = i === j;
        var base = diag ? C.green : C.red;
        var alpha = v === 0 ? 0.05 : 0.10 + 0.65 * (v / maxV);
        data.push({
          value: [j, i, v],
          itemStyle: {
            color: hexToRgba(base, alpha),
            borderColor: diag ? C.accent : C.splitLine,
            borderWidth: diag ? 1.2 : 0.5
          }
        });
      });
    });

    inst.setOption({
      backgroundColor: 'transparent',
      animationDuration: 240,
      grid: { left: 8, right: 14, top: 10, bottom: 4, containLabel: true },
      tooltip: Object.assign(tooltipBase(), {
        formatter: function (params) {
          var i = params.value[1], j = params.value[0], v = params.value[2];
          var total = matrix[i].reduce(function (a, b) { return a + b; }, 0) || 1;
          return '真实 ' + labels[i] + ' → 预测 ' + labels[j] +
            '<br/>数量：' + v + '（占该类 ' + (v / total * 100).toFixed(1) + '%）';
        }
      }),
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: confTick(),
        axisLine: { lineStyle: { color: C.axis } },
        axisTick: { show: false },
        splitArea: { show: false }
      },
      yAxis: {
        type: 'category',
        data: labels,
        inverse: true,
        axisLabel: confTick(),
        axisLine: { lineStyle: { color: C.axis } },
        axisTick: { show: false }
      },
      series: [{
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          color: cssVar('--text', C.text),
          fontSize: 11,
          fontFamily: MONO,
          formatter: function (p) { return String(p.value[2]); }
        },
        emphasis: { itemStyle: { borderColor: C.accent, borderWidth: 1.5 } }
      }]
    }, { notMerge: true });
  }

  /* ---------------- PCA 散点 ---------------- */

  /** pca：result.pca；test_points 圆环 + 误分类 ×。 */
  function renderPca(pca) {
    var inst = instances.pca;
    if (!inst) return;
    var points = pca && Array.isArray(pca.points) ? pca.points : [];
    var labels = pca && Array.isArray(pca.labels) ? pca.labels : [];
    if (!points.length || !labels.length) {
      inst.clear();
      setOverlay('pca', true);
      return;
    }
    setOverlay('pca', false);

    var testSet = {};
    var testPred = {};
    (pca.test_indices || []).forEach(function (idx, k) {
      testSet[idx] = true;
      testPred[idx] = (pca.test_predictions || [])[k];
    });

    var evr = pca.explained_variance_ratio || [];

    // 按类别分组散点
    var byClass = {};
    points.forEach(function (p, i) {
      var k = labels[i];
      (byClass[k] = byClass[k] || []).push({ p: p, idx: i });
    });

    var series = Object.keys(byClass).map(function (k) {
      var data = byClass[k].map(function (it) {
        return {
          value: [Number(it.p[0]), Number(it.p[1])],
          idx: it.idx,
          tested: !!testSet[it.idx],
          pred: testPred[it.idx]
        };
      });
      return {
        name: clsLabel(Number(k)),
        type: 'scatter',
        data: data,
        symbolSize: 6,
        z: 5,
        itemStyle: { color: global.classColor(Number(k)), opacity: 0.85 },
        emphasis: { scale: 1.4 }
      };
    });

    // 测试样本圆环
    var rings = [];
    points.forEach(function (p, i) {
      if (testSet[i]) rings.push([Number(p[0]), Number(p[1])]);
    });
    if (rings.length) {
      series.push({
        name: '测试样本',
        type: 'scatter',
        data: rings,
        symbol: 'circle',
        symbolSize: 10.5,
        silent: true,
        z: 8,
        itemStyle: { color: 'transparent', borderColor: C.text, borderWidth: 1.2, opacity: 0.9 }
      });
    }

    // 误分类 ×（自定义双线交叉）
    var misses = [];
    points.forEach(function (p, i) {
      if (testSet[i] && testPred[i] !== undefined && testPred[i] !== labels[i]) {
        misses.push([Number(p[0]), Number(p[1])]);
      }
    });
    if (misses.length) {
      series.push({
        name: '误分类',
        type: 'custom',
        data: misses,
        silent: true,
        z: 12,
        renderItem: function (params, api) {
          var p = api.coord([api.value(0), api.value(1)]);
          var r = 4.5;
          return {
            type: 'group',
            children: [
              { type: 'line', shape: { x1: p[0] - r, y1: p[1] - r, x2: p[0] + r, y2: p[1] + r }, style: { stroke: C.red, lineWidth: 2 } },
              { type: 'line', shape: { x1: p[0] + r, y1: p[1] - r, x2: p[0] - r, y2: p[1] + r }, style: { stroke: C.red, lineWidth: 2 } }
            ]
          };
        }
      });
    }

    inst.setOption({
      backgroundColor: 'transparent',
      animation: false,
      legend: {
        top: 2,
        left: 6,
        itemWidth: 12,
        itemHeight: 8,
        textStyle: { color: C.textDim, fontSize: 10 },
        data: Object.keys(byClass).map(function (k) { return clsLabel(Number(k)); })
      },
      grid: { left: 10, right: 18, top: 30, bottom: 22, containLabel: true },
      tooltip: Object.assign(tooltipBase(), {
        trigger: 'item',
        formatter: function (params) {
          if (params.seriesType === 'custom' || params.seriesName === '测试样本') return '';
          var d = params.data || {};
          var lines = ['#' + d.idx + ' · ' + params.seriesName + '（真实）',
            'PCA (' + fmtNum(d.value[0], 3) + ', ' + fmtNum(d.value[1], 3) + ')'];
          if (d.tested) {
            lines.push('测试样本 · 预测：' + clsLabel(d.pred) +
              (d.pred === labels[d.idx] ? ' ✔' : ' ✘ 误分类'));
          }
          return lines.join('<br/>');
        }
      }),
      xAxis: valueAxis('PC1 ' + UI.fmtPct(evr[0])),
      yAxis: Object.assign(valueAxis('PC2 ' + UI.fmtPct(evr[1])), { nameLocation: 'end', nameGap: 8 }),
      series: series
    }, { notMerge: true });
  }

  /* ---------------- 剪枝曲线（双 y 轴） ---------------- */

  /** compare：compare_variants 结果（curve + chosen_alpha）。 */
  function renderPruneCurve(compare) {
    var inst = instances.prune;
    if (!inst) return;
    var curve = compare && Array.isArray(compare.curve) ? compare.curve : [];
    if (!curve.length) { inst.clear(); return; }

    var alphas = curve.map(function (c) { return UI.fmtAlpha(c.alpha); });

    // 选中的 α 对应的曲线下标（取最接近者）
    var chosenIdx = -1;
    if (compare.chosen_alpha != null) {
      var best = Infinity;
      curve.forEach(function (c, i) {
        var d = Math.abs(c.alpha - compare.chosen_alpha);
        if (d < best) { best = d; chosenIdx = i; }
      });
    }

    var valSeries = {
      name: '验证准确率',
      type: 'line',
      yAxisIndex: 0,
      data: curve.map(function (c) { return Number(c.val.toFixed(4)); }),
      smooth: false,
      symbol: 'circle',
      symbolSize: 5.5,
      z: 5,
      lineStyle: { color: C.green, width: 2 },
      itemStyle: { color: C.green }
    };
    if (chosenIdx >= 0) {
      valSeries.markLine = {
        silent: true,
        symbol: 'none',
        lineStyle: { color: C.text, type: 'dashed', width: 1.2 },
        label: {
          color: C.textDim,
          fontSize: 10,
          fontFamily: MONO,
          formatter: 'α* ' + UI.fmtAlpha(compare.chosen_alpha)
        },
        data: [{ xAxis: chosenIdx }]
      };
    }

    inst.setOption({
      backgroundColor: 'transparent',
      animationDuration: 240,
      legend: {
        top: 2,
        right: 8,
        itemWidth: 14,
        itemHeight: 8,
        textStyle: { color: C.textDim, fontSize: 10 },
        data: ['训练准确率', '验证准确率', '叶节点数']
      },
      grid: { left: 10, right: 14, top: 32, bottom: 22, containLabel: true },
      tooltip: Object.assign(tooltipBase(), {
        trigger: 'axis',
        formatter: function (params) {
          var i = params[0].dataIndex;
          var c = curve[i] || {};
          var lines = ['<b>α = ' + UI.fmtAlpha(c.alpha) + '</b>' +
            (i === chosenIdx ? '　<span style="color:' + C.green + '">✔ 选中</span>' : ''),
            '训练准确率：' + UI.fmtPct(c.train),
            '验证准确率：' + UI.fmtPct(c.val),
            '叶节点：' + c.leaves + ' · 深度：' + c.depth + ' · 节点：' + c.nodes];
          return lines.join('<br/>');
        }
      }),
      xAxis: {
        type: 'category',
        name: 'α',
        nameLocation: 'middle',
        nameGap: 24,
        data: alphas,
        nameTextStyle: { color: C.textFaint, fontSize: 10 },
        axisLabel: Object.assign(axisLabelStyle(), { fontFamily: MONO, rotate: alphas.length > 6 ? 22 : 0 }),
        axisLine: { lineStyle: { color: C.axis } },
        axisTick: { show: false }
      },
      yAxis: [
        {
          type: 'value',
          name: '准确率',
          scale: true,
          nameTextStyle: { color: C.textFaint, fontSize: 10 },
          axisLabel: { color: C.textDim, fontSize: 10, formatter: function (v) { return UI.fmtPct(v, 0); } },
          axisLine: { show: false },
          splitLine: { lineStyle: { color: C.splitLine } }
        },
        {
          type: 'value',
          name: '叶节点数',
          min: 0,
          minInterval: 1,
          nameTextStyle: { color: C.textFaint, fontSize: 10 },
          axisLabel: { color: C.textDim, fontSize: 10 },
          axisLine: { show: false },
          splitLine: { show: false }
        }
      ],
      series: [
        {
          name: '叶节点数',
          type: 'bar',
          yAxisIndex: 1,
          barWidth: '42%',
          data: curve.map(function (c) { return c.leaves; }),
          itemStyle: { color: hexToRgba(C.bar, 0.45), borderRadius: 0 },
          label: {
            show: true,
            position: 'top',
            color: C.textFaint,
            fontSize: 9.5,
            fontFamily: MONO
          }
        },
        {
          name: '训练准确率',
          type: 'line',
          yAxisIndex: 0,
          data: curve.map(function (c) { return Number(c.train.toFixed(4)); }),
          smooth: false,
          symbol: 'emptyCircle',
          symbolSize: 5,
          z: 4,
          lineStyle: { color: C.textDim, width: 1.5, type: 'dashed' },
          itemStyle: { color: C.textDim }
        },
        valSeries
      ]
    }, { notMerge: true });
  }

  /* ---------------- 导出 / 其他 ---------------- */

  /** 取图表 PNG dataURL；pixelRatio=2，底色在调用时按当前主题读取。 */
  function dataURL(name) {
    var inst = instances[name];
    if (!inst) return null;
    var light = isLight();
    var bg = cssVar('--card', light ? '#ffffff' : '#141414') || (light ? '#ffffff' : '#141414');
    try {
      return inst.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: bg });
    } catch (e) {
      return null;
    }
  }

  function resizeAll() {
    Object.keys(instances).forEach(function (k) {
      try { instances[k].resize(); } catch (e) { /* 容器不可见时忽略 */ }
    });
  }

  /** 清空主视图四图并显示空态覆盖层。 */
  function clearMain() {
    ['tree', 'candidates', 'confusion', 'pca'].forEach(function (k) {
      var inst = instances[k];
      if (inst) {
        try { inst.clear(); } catch (e) { /* noop */ }
      }
      setOverlay(k, true);
    });
  }

  global.Charts = {
    loadECharts: loadECharts,
    initCharts: initCharts,
    ensureCompareCharts: ensureCompareCharts,
    setDatasetInfo: setDatasetInfo,
    renderTree: renderTree,
    renderCandidates: renderCandidates,
    renderConfusion: renderConfusion,
    renderPca: renderPca,
    renderPruneCurve: renderPruneCurve,
    bindTreeClick: bindTreeClick,
    dataURL: dataURL,
    resizeAll: resizeAll,
    refreshTheme: refreshTheme,
    clearMain: clearMain
  };
})(window);
