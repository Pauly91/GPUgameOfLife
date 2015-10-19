import os

iterations = input("iterations:")
dataSet = raw_input("data Set:")

os.system("./gpu "+dataSet+" "+str(iterations))
os.system("./gpuChar "+dataSet+" "+str(iterations))

<<<<<<< HEAD
os.system("cmp --silent GPUOutput GPUOutputChar && echo '### SUCCESS: Files Are Identical! ###' || echo '### WARNING: Files Are Different! ###'")
=======
os.system("cmp --silent GPUOutput CPUOutput && echo '### SUCCESS: Files Are Identical! ###' || echo '### WARNING: Files Are Different! ###'")
>>>>>>> 5d77fde5af2553e2cc7c10a1a4f8c31c0b0fedba
