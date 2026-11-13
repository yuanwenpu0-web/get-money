# P18 数值修复后端加载确认（工程验证，非 edge 判定）

生成：2026-11-13T01:53:17+00:00；工程状态 PASS。

用户明确授权停止并重启 QuantDesk-GetMoney 后，重启已执行，HTTP健康接口返回ok=true。
当前服务mode=live，exchange_env=demo，保留原有执行模式。

监听进程PID=3296，进程启动毫秒=1794534584919，任务启动毫秒=1794534584000；进程启动晚于源码修复。
当前源码SHA256与80币诊断报告中的指纹一致。

此前诊断报告中的BLOCKED_APPROVAL是当时审批状态，本报告记录随后明确授权与重启验证。

## 边界与启用决策

窗口为部署检查时点，币池/绑定均未改，default_enabled不变。
成本仍以名义本金往返0.12%为统一参照；本轮不模拟交易，CI置信区间与随机基线均NOT_APPLICABLE/NOT_RUN。
收益verdict为inconclusive（未评估），工程健康检查通过不解除P18-002方法门槛或独立OOS要求。

## 测试证据

- `output/fit-log-focused-pytest.log`：68 passed in 1.20s
- `output/fit-log-full-pytest.log`：3 failed, 2152 passed, 5 warnings in 94.82s (0:01:34)
