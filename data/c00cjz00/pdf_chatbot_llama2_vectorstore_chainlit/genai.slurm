#!/work/u00cjz00/binary/bash5.0/bin/bash
#SBATCH -A MST110386                                                    ### project number, Example MST109178
#SBATCH -J _t2g_							                            ### Job name, Exmaple jupyterlab
#SBATCH -p gp4d                                                         ### Partition Name, Example ngs1gpu
#SBATCH --nodes=1                                                       ### Nodes, Default 1, node number
#SBATCH --ntasks-per-node=1                                             ### Tasks, Default 1, per node tasks
#SBATCH -c 4                                                            ### Cores assigned to each task, Example 4
#SBATCH --gres=gpu:1                                                    ### GPU number, Example gpu:1
#SBATCH --time=0-1:00:00                                                ### Runnung time, days-hours:minutes:seconds or hours:minutes:seconds
#SBATCH -o genai_%j.out     											### Log folder, Here %j is job ID
#SBATCH -e genai_%j.err     											### Log folder, Here %j is job ID

## 帳號
iam=$(whoami)
charbot_dir=/work/${iam}/chainlit_demo
home_dir=${charbot_dir}/home
tmp_dir=${charbot_dir}/tmp
mkdir -p ${home_dir} ${tmp_dir}
rsync -avHS /work/u00cjz00/slurm_jobs/github/pdf_chatbot_llama2_vectorstore_chainlit/ ${charbot_dir}/home/pdf_chatbot_llama2_vectorstore_chainlit


# IMAGE
IMAGE="/work/u00cjz00/nvidia/pytorch_2.1.0-cuda11.8-cudnn8-devel.sif"

# NODE
noed_hostname=$(hostname -s)
noed_port=$(python3 /work/u00cjz00/binary/availablePort.py)
node_ip=$(cat /etc/hosts |grep "$(hostname -a)" | awk '{print $1}')

# message
echo ""
echo ""
echo "****************************** 請輸入下方指令 ******************************"
echo ""
echo "# STEP1: Execute cmd in your client below "
echo ssh -L ${noed_port}:${noed_hostname}:${noed_port} ${iam}@ln01.twcc.ai
echo ""
echo "# STEP2: Open url below "
echo "http://localhost:${noed_port}/"
echo ""
echo ""
echo "***********************************************************************************************"

# Singuilarty
ps aux | grep chainlit | awk '{print $2}' | xargs kill -9 
ml libs/singularity/3.10.2
singularity exec --nv --no-home -B ${home_dir}:/home -B ${tmp_dir}:/tmp -B /work ${IMAGE} bash -c "cd /home/pdf_chatbot_llama2_vectorstore_chainlit && pip install -r requirements.txt -q"
singularity exec --nv --no-home -B ${home_dir}:/home -B ${tmp_dir}:/tmp -B /work ${IMAGE} bash -c "cd /home/pdf_chatbot_llama2_vectorstore_chainlit &&  pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ -q"
singularity exec --nv --no-home -B ${home_dir}:/home -B ${tmp_dir}:/tmp -B /work ${IMAGE} bash -c "cd /home/pdf_chatbot_llama2_vectorstore_chainlit && python3 ingest.py"
singularity exec --nv --no-home -B ${home_dir}:/home -B ${tmp_dir}:/tmp -B /work ${IMAGE} bash -c "cd /home/pdf_chatbot_llama2_vectorstore_chainlit && ~/.local/bin/chainlit run model_gptq.py --port ${noed_port} --host ${noed_hostname}" &

## Get pid
pid0=$!
sleep 2
pid1=$(pgrep -P ${pid0})
pid=${pid1}
wait $pid0


