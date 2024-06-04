#openssh daemon
#启动/start
sshd
#关闭/close
lsof -i | grep sshd | awk '{print $1}' | xargs kill
#或者
pkill sshd
#使用密码登录
pkg up
pkg install openssh
#然后
passwd
#这时会让你输密码，自己编一个
#登录时的用户名可以用whoami这个命令看到。



#nodejs-package-manager(npm)
#fix bug
vi $PREFIX/lib/node_modules/npm/node_modules/worker-farm/lib/farm.js
 
#将'maxConcurrentWorkers        : (require('os').cpus() || { length: 1 }).length'内的1改动一下，改成一个小于cpu核心数的数字。


#Aria2
#Rpc 服务
aria2c --enable-rpc --rpc-listen-all
#然后可以用transdroid从127.0.0.1:6800连接



#lftp
#使用以下命令登录ftp服务器：
lftp ftp://用户名[:密码]@服务器地址[:端口]
#标准方式，推荐
#如果不指定端口，默认21端口
#如果不在命令中使用明文输入密码，连接时会询问密码(推荐)
#可以使用“书签”收藏服务器站点，也可以在lftp中为当前站点定义别名：
lftp >bookmark           #显示所有收藏
lftp >bookmark add <别名>  #使用 别名 收藏当前站点
#使用别名登录 ftp服务器：
lftp <别名>
#文件下载
#单个文件
lftp >get <name>	 
#目录	
lftp >mirror <dirname>
#windows支持
lftp >set ftp:charset gbk   #设置远程编码为gbk
lftp >set file:charset utf8 #设置本地编码(Linux系统默认使用 UTF-8，这一步通常可以省略)  
 

#Python
#Helloworld
python -c "print('Hello,world.')"
#This
python -c "import this"



#X11
#GUI环境
#先开启x11仓库
pkg install x11-repo
#tigervnc
pkg install tigervnc; vncserver
#会让你输入密码，编一个:D
#bash,zsh环境变量
export DISPLAY=":1"
#fish
set -x DISPLAY ":1"
#默认5901端口
#选一个窗口管理器
#fluxbox
pkg install fluxbox
WRITE_SCRIPT() {
cat > ~/.vnc/xstartup <<EOF
#!/data/data/com.termux/files/usr/bin/sh
fluxbox-generate_menu
fluxbox &
EOF
}
WRITE_SCRIPT
#fish没有here文档，暂时切到bash下吧。


#bettercap
#运行
[ -d /system/bin ]&&export PATH=$PATH:/system/bin:/sbin
bettercap
