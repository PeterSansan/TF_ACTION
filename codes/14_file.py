# -*- coding: utf-8 -*- 

import sys

f=open('a.txt','a+') # 追加

old=sys.stdout #将当前系统输出储存到一个临时变量中
sys.stdout=f  #输出重定向到文件
print 'Hello weird' #测试一个打印输出
sys.stdout.flush() # 刷新文件流
sys.stdout=old #还原原系统输出
f.close() 
print open('a.txt','r').read()

# 第二种方法
# f=open('test.txt','a+')
# s= '123'
# abc= '456'
# print >> f, s,abc
# f.close()