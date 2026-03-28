#!/bin/bash
#SBATCH -J test                                ## 作业名
#SBATCH -p ksagexclu01                         ## 队列（按截图）
#SBATCH -N 1                                   ## 申请计算节点数
#SBATCH --ntasks-per-node=1                    ## 每节点进程数
#SBATCH --exclusive                            ## 独占节点
##SBATCH --constraint=32core                   ## 申请到 32 核节点时可取消注释
#SBATCH --cpus-per-task=8                      ## 每个任务使用 8 个 CPU 核心
#SBATCH --gres=gpu:1                           ## 单节点 1 张 GPU
##SBATCH -x j17r2n04                           ## 不分配的计算节点
##SBATCH -w e10r4n02                           ## 指定节点运行
#SBATCH -n 1                                   ## 总任务数
#SBATCH -o out.%j                              ## 标准输出
#SBATCH -e err.%j                              ## 错误日志
#SBATCH --time=01:00:00                        ## 最长运行时间 1 小时

#### 加载环境
source ~/.bashrc
# 如果你使用 conda，请取消下面两行注释并改成自己的环境名
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

#### 进入项目目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

#### 运行命令
python "$SCRIPT_DIR/main.py" \
  -data UNSW \
  -algo FIDSUS \
  -nc 50 \
  -gr 100 \
  -ls 1 \
  -dev cuda \
  -did 0
