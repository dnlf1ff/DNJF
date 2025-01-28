#!/bin/bash

mlp_out="mlp.list"
echo "mlp" > $mlp_out

for dir in */; do
    if [ -d "$dir" ]; then
        echo "$dir" >> $mlp_out
    fi
done


