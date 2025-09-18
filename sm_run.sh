# SASRec
python run_recbole.py --model=SASRec --dataset=beauty --gpu_id=6
python run_recbole.py --model=SASRec --dataset=sports --gpu_id=6
python run_recbole.py --model=SASRec --dataset=toys --gpu_id=7

# SASRec-LLM2X
python run_recbole.py --model=LLM2X_SASRec --dataset=beauty --gpu_id=6 &&
python run_recbole.py --model=LLM2X_SASRec --dataset=sports --gpu_id=6 &&
python run_recbole.py --model=LLM2X_SASRec --dataset=toys --gpu_id=6

# SASRec-AlphaRec
python run_recbole.py --model=AlphaRec_SASRec --dataset=beauty --gpu_id=7 &&
python run_recbole.py --model=AlphaRec_SASRec --dataset=sports --gpu_id=7 &&
python run_recbole.py --model=AlphaRec_SASRec --dataset=toys --gpu_id=7

# generate multimodal embeddings with GME Qwen2VL-2B-Instruct
python gme_qwen2vl.py --dataset=beauty --gpu_id=6 --embedding_encoder=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
python gme_qwen2vl.py --dataset=sports --gpu_id=7 --embedding_encoder=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
python gme_qwen2vl.py --dataset=toys --gpu_id=7 --embedding_encoder=Alibaba-NLP/gme-Qwen2-VL-2B-Instruct

# generate multimodal embeddings with GME Qwen2VL-7B-Instruct
python gme_qwen2vl.py --dataset=beauty --gpu_id=6 --embedding_encoder=Alibaba-NLP/gme-Qwen2-VL-7B-Instruct --batch_size=16 &&
python gme_qwen2vl.py --dataset=toys --gpu_id=6 --embedding_encoder=Alibaba-NLP/gme-Qwen2-VL-7B-Instruct --batch_size=16

python gme_qwen2vl.py --dataset=sports --gpu_id=7 --embedding_encoder=Alibaba-NLP/gme-Qwen2-VL-7B-Instruct --batch_size=16



# SASRec-LLM2X with GME Qwen2VL-2B-Instruct
python run_recbole.py --model=LLM2X_SASRec --dataset=beauty --text_encoder=gme_qwen2vl2b_text_fp16 --gpu_id=7 &&
python run_recbole.py --model=LLM2X_SASRec --dataset=beauty --text_encoder=gme_qwen2vl2b_image_fp16 --gpu_id=7

# SASRec-LLM2X with GME Qwen2VL-2B-Instruct
python run_recbole.py --model=AlphaRec_SASRec --dataset=beauty --text_encoder=gme_qwen2vl2b_text_fp16 --gpu_id=7 &&
python run_recbole.py --model=AlphaRec_SASRec --dataset=beauty --text_encoder=gme_qwen2vl2b_image_fp16 --gpu_id=7


