VLM Agent 排版规划有效性 (Table 1)

   配置          方法描述                     评价指标
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Center Fixed  固定规则布局（中心区域等分   Mean IoU ↑, Median IoU ↑
   w/o Agent     随机采样bbox               Mean IoU ↑, Median IoU ↑
   w/o Grid      VLM 规划但无网格坐标         Mean IoU ↑, Median IoU ↑
   Grid 3×3      VLM + 3×3 Grid Overlay     Mean IoU ↑, Median IoU ↑
   Grid 5×5      VLM + 5×5 Grid Overlay     Mean IoU ↑, Median IoU ↑  (最佳)
   Grid 8×8      VLM + 8×8 Grid Overlay     Mean IoU ↑, Median IoU ↑

实验结果:
- Baseline (Fixed Center): 0.1855 Mean IoU
- w/o Agent (Random): 0.2147 Mean IoU (+15.8%)
- w/o Grid (VLM only): 0.2703 Mean IoU (+45.7%)
- VLM + 3×3 Grid: 0.4475 Mean IoU (+141.3%)
- VLM + 5×5 Grid: 0.5531 Mean IoU (+198.2%) ← 最佳
- VLM + 8×8 Grid: 0.3776 Mean IoU (+103.6%)

关键发现:
1. Grid 引导显著提升了 VLM 的定位精度 (103.6% - 198.2% 提升)
2. 5×5 Grid 表现最佳，在坐标精度和可读性之间达到最佳平衡
3. 8×8 网格表现下降，说明过密的网格会引入视觉噪声，降低 VLM 的理解能力
4. 中等密度网格 (5×5) 优于过低 (3×3) 或过高密度 (8×8)

实验实现:
'/Users/yanzexuan/code/Calligrapher/ablation/table1_vlm_localization.py'
- 读取公式来自: /Users/yanzexuan/code/Calligrapher/eval/UnseenWords/unseen_mid_sci.jsonl
- 渲染使用: infer/formula_helper.py
- Grid 渲染使用: infer/VLM_agent.py 中的 _add_grid_overlay 函数
