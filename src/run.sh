#!/bin/bash

# defined commands array
commands=(
    "python interact.py  --model Qwen2.5-7B-Instruct"
    "python interact.py  --model qwen3-8b"
    "python interact.py  --model deepseek-v3"
    "python interact.py  --model deepseek-r1"
    "python interact.py  --model gpt-4.1-mini"
    "python interact.py  --model gpt-4o"
)

# execute each command in the array
for command in "${commands[@]}"; do
    echo "Running: $command"
    eval $command
    # check if command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred while running: $command"
        exit 1
    fi
    echo "Finished: $command"
    echo "-----------------------------------"
done

echo "All commands executed successfully."
