

all: 
	make -C src/proto
	make -C src/qt/ui


ultra: 
	scp -r src/proto root@192.168.3.111:/home/root/Projects/src/
	scp -r src/remote/ultra96 root@192.168.3.111:/home/root/Projects/src/remote
	scp -r scripts/ root@192.168.3.111:/home/root/Projects/
	scp -r src/utils root@192.168.3.111:/home/root/Projects/src/
	scp -r src/config root@192.168.3.111:/home/root/Projects/src/

ultra_py:
	scp -r src/remote/ultra96 root@192.168.3.111:/home/root/Projects/src/remote/