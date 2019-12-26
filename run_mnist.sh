echo "===== Teacher ====="
python train_teacher.py --batch_size 256 --epochs 10 --lr 0.01 --dataset mnist --model lenet5 --weight_decay 1e-4 # --verbose

echo "===== DFAD ====="
python DFAD_mnist.py --ckpt checkpoint/teacher/mnist-lenet5.pt # --verbose