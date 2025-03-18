# Voila
A speech role-play model.

## Inference
### Inference scipt
```shell
for model_name in "maitrix-org/Voila-audio-alpha" "maitrix-org/Voila-base" "maitrix-org/Voila-chat"; do
    # Text chat
    python infer.py \
        --model-name ${model_name} \
	    --instruction "" \
	    --input-text "Hello" \
	    --task-type chat_tito
    # Voice chat
    python infer.py \
        --model-name ${model_name} \
	    --instruction "" \
	    --input-audio "examples/test1.mp3" \
	    --task-type chat_aiao
done

# Autonomous mode
python infer.py \
    --model-name "maitrix-org/Voila-autonomous-preview" \
	--instruction "" \
	--input-audio "examples/test_autonomous1.mp3" \
	--task-type chat_aiao_auto
```

### Gradio demo
```shell
python gradio_demo.py
```

## Citation
```bib
```
