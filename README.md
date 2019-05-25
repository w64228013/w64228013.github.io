
本博客参考[原仓库地址](https://github.com/Huxpro/huxpro.github.io)搭建而成．
使用此仓库构建个人博客步骤如下：
1. 新建仓库
2. clone本仓库并修改
3. 上传检验

## 新建仓库
新建仓库，名称为"username.github.io".比如这个仓库的名字"CNyuzhang.github.io"．然后将[本仓库](https://github.com/CNyuzhang/CNyuzhang.github.io)下载到本地.

## 修改信息
#### 网站架构
![](/img/README/paper.png)
这里只需要了解几个文件就行了．
* _config.yml　　这是全局配置文件
* _drafts　　　　存放草稿的文件夹
* _includes　　　设置页面底部和头部文件的文件夹
* _layouts　　　设置页面布局的配置文件 
* _posts　　　　放置博客文章的文件夹
* img　　　　　放置图片的文件夹

详细的信息[参考这里](https://www.jekyll.com.cn/docs/structure/).

#### 修改文件
###### 1. 更改＿config.yml
![](/img/README/config.png)
> 首先修改博客名称，个人邮箱主页等信息，header-img表示首页背景图．然后文件最后可以添加朋友链接．主页图片如下：

![](/img/README/home-page.png)

###### 2. 修改about.html
about界面如下所示：
![](/img/README/about-page.png).
对应的文字更改在about.html 文件的相应位置：
![](/img/README/about.png)

###### 3. 更改default.html
![](/img/README/little.png)
img src="/img/apple-touch-icon.png"，这个地址的图片表示的是网站的缩略图，更改为你喜欢的图片即可．
![](/img/README/picture.png)

## 上传完成
将上面修改好信息后的仓库上传到你的仓库中，稍等两分钟，进入网址＂youname.github.io"即可看到你的博客页面．查看博客主页若有不满意地方就重新更改后提交．


---
## 创建博客
每一篇文章文件命名采用的是**2019-05-23-Hello-2019.md**时间+标题的形式，空格用-替换连接。
文件的格式是 .md 的 [MarkDown](https://zh.wikipedia.org/wiki/Markdown) 文件。
我们的博客文章格式采用是 **MarkDown+ YAML** 的方式。
YAML 就是我们配置 _config文件用的语言。
MarkDown 是一种轻量级的「标记语言」，很简单。[花半个小时看一下](https://sspai.com/post/25137)就能熟练使用了
大概就是这么一个结构

```
---
layout:     post                    # 使用的布局（不需要改）
title:      My First Post           # 标题 
subtitle:   Hello World, Hello Blog #副标题
date:       201９-05-23         　   # 时间
author:     CHEETAHER               # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - tags
---
## Hey
>这是我的第一篇博客。
```
进入你的博客主页，新的文章将会出现在你的主页上.

按照格式创建完成提交后即可在主页看到博客．



## 相关内容
* [使用git进行提交与更改](http://blog.jobbole.com/53573/)
* 网站流量统计．使用[revolvermaps](https://www.revolvermaps.com/)进行统计．将订阅revolvermaps的代码放在page.html文件的最后，如图位置:
![](/img/README/vistor.png)
效果如下：
![](/img/README/views.png)

* 评论系统使用[来必力](http://www.laibili.com.cn/)．分别放置在博客页面（＿layouts/post.html）和about（about.html）页面．
![](/img/README/laibili-post.png)
![](/img/README/laibili-about.png)

效果如下：
![](/img/README/laibili.png)

