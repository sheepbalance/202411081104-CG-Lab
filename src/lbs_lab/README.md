# 202411081104 杨衡 人工智能

<!--[程序执行指令](uv run -m src.lbs_lab.run_lbs_lab  --model-dir ./models --out-dir ./outputs --joint-id 2 --anim-joint-id 18 --anim-end-angle -90.0)--> 
# 实验8执行结果
## 修改教程范例新增输出gif，新增2个执行参数数，1.指定动画转动关节 2.从零度转动至最后角度
### 输出执行指令 uv run -m src.lbs_lab.run_lbs_lab  --model-dir ./models --out-dir ./outputs --joint-id 2 --anim-joint-id 18 --anim-end-angle -90.0
![这是图片1](./outputs/stage_a_template_weights.png "实验8图_stage_a_template_weights.png:展示模板网格以及某一个指定关节的权重热力图")
![这是图片2](./outputs/all_joint_weights.png "实验8图_all_joint_weights.png:展示整个人体表面由哪些关节主导控制")
![这是图片3](./outputs/stage_b_shaped_joints.png "实验8图_stage_b_shaped_joints.png:人体体型已经变化，关节点叠加在身体内部合理位置")
![这是图片4](./outputs/stage_c_pose_offsets.png "实验8图_stage_c_pose_offsets.png:能看出姿态相关校正主要集中在发生弯曲的部位附近")
![这是图片5](./outputs/stage_d_lbs_result.png "实验8图_stage_d_lbs_result.png:人体已经进入最终姿态")
![这是图片6](./outputs/comparison_grid.png "实验8图_comparison_grid.png:四个阶段之间的区别一目了然")
[实验8_summary.txt:记录模型基础信息以及手写 LBS 与官方前向结果的误差](./outputs/summary.txt "实验8_summary.txt")
## 实验选作内容执行结果
![这是图片7](./outputs/lbs_animation.gif "实验8图_lbs_animation.gif:「平滑过度」（Smooth Blending）")
