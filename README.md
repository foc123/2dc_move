# 2dc_move

输入为VDI分布，每一次的输出动作为两个节点的动作，episodes为8000

loss function在初始阶段快速下降，但随后持续波动，未能收敛到0
![loss_function](https://user-images.githubusercontent.com/89006608/187025921-d56684e6-8dbc-449d-a77a-e22d68d90392.png)

reward定义为 （targetVDI  -  当前VDI）/targetVDI，目标VDI为1e-10
![reward](https://user-images.githubusercontent.com/89006608/187025976-3240fb25-febe-4750-93d5-28358fda3897.png)

cap_location代表decap位置的序号，范围为【0，143】共144个点，实际上decap位置有变化而图中为一条直线
![cap_location_per_eps](https://user-images.githubusercontent.com/89006608/187026209-61c1bd0a-3ce6-4835-a2f8-2da21532248e.png)
