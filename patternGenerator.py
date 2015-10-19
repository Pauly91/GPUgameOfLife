from random import randint

height = input("Enter the height:")
width = input("Enter the width:")
<<<<<<< HEAD
fp = open('data2','w')
=======
fp = open('pattern','w')
>>>>>>> 5d77fde5af2553e2cc7c10a1a4f8c31c0b0fedba
fp.write(str(height) + '\n' + str(width) + '\n')
try:
	for i in range(0,height):
		for j in range(0,width):
			fp.write(str(randint(0,1)) + " ")
	fp.write('\n')
	fp.close();
	exit(0)
except KeyboardInterrupt:
	fp.close()
	exit(1)






