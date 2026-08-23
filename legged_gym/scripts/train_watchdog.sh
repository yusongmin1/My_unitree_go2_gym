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
TASK=${1:?用法: bash train_watchdog.sh <task|--task=xxx> [train.py其它参数...]}; shift
# 兼容 --task=xxx 与 --task xxx 两种写法
TASK=${TASK#--task=}
if [ "$TASK" = "--task" ] || [ "$TASK" = "--task=" ]; then
    TASK=${1:?缺少任务名}; shift
fi

# numpy/OpenBLAS 偶发段错误缓解
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/zju/miniconda3/envs/amp/lib:${LD_LIBRARY_PATH:-}
PY=/home/zju/miniconda3/envs/amp/bin/python
cd "$(dirname "$0")/../.."   # 回到仓库根

# 避开崩溃高发核心：历次段错误均落在逻辑 CPU 10-11（物理 core 20，含超线程兄弟）
if [ -z "${TASKSET_CMD:-}" ]; then
    TOTAL_CPU=$(nproc)
    EXCLUDE="10 11"
    ALLOWED=""
    for i in $(seq 0 $((TOTAL_CPU-1))); do
        skip=0
        for e in $EXCLUDE; do [ "$i" = "$e" ] && skip=1; done
        [ $skip -eq 0 ] && ALLOWED="$ALLOWED,$i"
    done
    TASKSET_CMD="taskset -c ${ALLOWED#,}"
    echo "[watchdog] 避开核心 10,11（core 20），使用: $TASKSET_CMD"
fi

RESUME=""
ROUND=0
while true; do
    ROUND=$((ROUND+1))
    echo "[watchdog] 第 $ROUND 次启动: train.py --task $TASK $RESUME $*"
    $TASKSET_CMD $PY legged_gym/scripts/train.py --task "$TASK" $RESUME "$@"
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "[watchdog] 训练正常结束（迭代数达标）。"
        break
    fi
    if [ $CODE -eq 2 ]; then
        echo "[watchdog] 参数错误（argparse 用法错误），重试无意义，退出。请检查命令行参数。"
        break
    fi
    echo "[watchdog] train.py 异常退出（码 $CODE，139=段错误），10 秒后自动 --resume 重启..."
    sleep 10
    RESUME="--resume"
done
