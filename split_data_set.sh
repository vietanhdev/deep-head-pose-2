#!/bin/bash
cd  data
mkdir train_set
mkdir val_set

echo "Making validation set"
for i in {0..2000}
do
   mv 300W_LP_prepared/"$i".png val_set/300W_LP-"$i".png
   mv 300W_LP_prepared/"$i".json val_set/300W_LP-"$i".json
   mv BIWI_prepared/"$i".png val_set/BIWI-"$i".png
   mv BIWI_prepared/"$i".json val_set/BIWI-"$i".json
done
echo "Making training set"
echo "300W_LP..."
for i in {2001..61224}
do
    mv 300W_LP_prepared/"$i".png train_set/300W_LP-"$i".png
    mv 300W_LP_prepared/"$i".json train_set/300W_LP-"$i".json
done
echo "BIWI..."
for i in {2001..15545}
do
    mv BIWI_prepared/"$i".png train_set/BIWI-"$i".png
    mv BIWI_prepared/"$i".json train_set/BIWI-"$i".json
done