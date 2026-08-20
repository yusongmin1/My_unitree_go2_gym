#!/bin/bash
# 训练看门狗：崩溃后自动 --resume 续训，直到正常跑完 max_iterations。
# 同时固定单线程 BLAS，规避 numpy 偶发段错误（_multiarray_umath segfault）。
#
# 用法：
#   bash train_watchdog.sh <task> [train.py 其它参数...]
# 例：
#   bash train_watchdog.sh go2_amp_cts --headless
#   bash train_watchdog.sh go2_stairs_dreamwaq --headless
set -u
TASK=${1:?用法: bash train_watchdog.sh <task> [train.py其它参数...]}; shift

# numpy/OpenBLAS 偶发段错误缓解
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/zju/miniconda3/envs/amp/lib:${LD_LIBRARY_PATH:-}
PY=/home/zju/miniconda3/envs/amp/bin/python
cd "$(dirname "$0")/../.."   # 回到仓库根

RESUME=""
ROUND=0
while true; do
    ROUND=$((ROUND+1))
    echo "[watchdog] 第 $ROUND 次启动: train.py --task $TASK $RESUME $*"
    $PY legged_gym/scripts/train.py --task "$TASK" $RESUME "$@"
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "[watchdog] 训练正常结束（迭代数达标）。"
        break
    fi
    echo "[watchdog] train.py 异常退出（码 $CODE，139=段错误），10 秒后自动 --resume 重启..."
    sleep 10
    RESUME="--resume"
done
