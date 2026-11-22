/* ============================================================
 * CARTLab — 全局状态与事件总线（state.js）
 *  - State：数据集信息、训练配置、运行结果、当前帧、选中节点、
 *    选中样本、对比结果；
 *  - Bus：轻量发布订阅，串联时间轴 / 图表 / 面板；
 *  - classPalette：三类样本的颜色（随明暗主题切换）。
 * ============================================================ */
(function (global) {
  'use strict';

  var CLASS_PALETTE_DARK = ['#8fbf7f', '#d8b26a', '#c98f8f'];
  var CLASS_PALETTE_LIGHT = ['#3d7a45', '#8a6a2f', '#a33a3a'];

  /** 当前主题下的类别颜色（按类别下标）。 */
  function classPalette() {
    return document.documentElement.dataset.theme === 'light'
      ? CLASS_PALETTE_LIGHT : CLASS_PALETTE_DARK;
  }

  function classColor(i) {
    var p = classPalette();
    return p[i % p.length];
  }

  var State = {
    datasetId: null,      // 当前数据集 id（iris）
    datasetInfo: null,    // get_dataset_info 结果
    result: null,         // run_training 完整结果
    frameIndex: 0,        // 当前帧下标
    track: 'build',       // 时间轴轨道：'build' 建树帧 | 'prune' 剪枝序列
    buildTree: null,      // 剪枝前的完整建树（prune.steps[0].tree），用于查已被剪掉的节点
    selectedNodeId: null, // 选中的节点 id（树 / 节点详情联动）
    sampleIndex: 0,       // 推理路径选中的样本下标（explanation 数组内）
    compare: null,        // compare_variants 结果
    running: false,
    comparing: false
  };

  var listeners = {};
  var Bus = {
    on: function (ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    emit: function (ev, payload) {
      (listeners[ev] || []).slice().forEach(function (fn) {
        try { fn(payload); } catch (e) { console.error('[Bus:' + ev + ']', e); }
      });
    }
  };

  /** 应用一次完整训练结果，重置帧游标 / 选中节点 / 选中样本。 */
  State.applyResult = function (result) {
    State.result = result || null;
    State.frameIndex = 0;
    State.track = 'build';
    State.selectedNodeId = null;
    State.sampleIndex = 0;
    // 剪枝序列首项是未剪枝的完整建树（后端 prune_sequence 保证），用于查已被剪掉的节点
    State.buildTree = (result && result.prune && Array.isArray(result.prune.steps) &&
      result.prune.steps[0] && result.prune.steps[0].tree) || null;
  };

  /** 在扁平节点列表（最终模型）中查找节点；找不到返回 null。 */
  State.findNode = function (nodeId) {
    if (!State.result || !Array.isArray(State.result.nodes)) return null;
    var hit = null;
    State.result.nodes.forEach(function (n) {
      if (String(n.id) === String(nodeId)) hit = n;
    });
    return hit;
  };

  /** 在剪枝前的完整建树中查找节点（含已被剪掉的）；找不到返回 null。 */
  State.findBuildNode = function (nodeId) {
    var root = State.buildTree;
    if (!root) return null;
    function walk(n) {
      if (!n) return null;
      if (String(n.id) === String(nodeId)) return n;
      var ch = n.children || [];
      for (var i = 0; i < ch.length; i++) {
        var hit = walk(ch[i]);
        if (hit) return hit;
      }
      return null;
    }
    return walk(root);
  };

  global.State = State;
  global.Bus = Bus;
  global.classPalette = classPalette;
  global.classColor = classColor;
})(window);
