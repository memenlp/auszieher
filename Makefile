

#================================================================
#   Copyright (C) 2022 Fisher. All rights reserved.
#   
#   文件名称：Makefile
#   创 建 者：YuLianghua
#   创建日期：2022年05月20日
#   描    述：
#
#================================================================

all: build

build:
	docker build . -f=Dockerfile -t=auszieher
	docker tag auszieher registry.shdocker.tuya-inc.top/ai-platform-public/auszieher:v0.1.6
	docker push registry.shdocker.tuya-inc.top/ai-platform-public/auszieher:v0.1.6

clean:
	echo y | docker system prune
