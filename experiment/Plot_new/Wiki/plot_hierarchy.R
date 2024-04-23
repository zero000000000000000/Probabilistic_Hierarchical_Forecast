# Libraries
#install.packages('ggraph')
library(ggraph)
library(igraph)
library(tidyverse)
library(RColorBrewer) 
# # create a data frame giving the hierarchical structure of your individuals
# d1=data.frame(from="origin", to=paste("group", seq(1,10), sep=""))
# d2=data.frame(from=rep(d1$to, each=10), to=paste("subgroup", seq(1,100), sep="_"))
# edges=rbind(d1, d2)

library(jsonlite)
e<-fromJSON('D:\\HierarchicalCode\\experiment\\Plot_new\\Tourism\\Hierarchy_info.json')
edge<-matrix(e,ncol=2,byrow=TRUE)
edge<-as.data.frame(edge)
names(edge)<-c('from','to')

# create a vertices data.frame. One line per object of our hierarchy
vertices = data.frame(
  name = unique(c(as.character(edge$from), as.character(edge$to)))
) 
# Let's add a column with the group of each name. It will be useful later to color points
vertices$group = edge$from[ match( vertices$name, edge$to ) ]


#Let's add information concerning the label we are going to add: angle, horizontal adjustement and potential flip
#calculate the ANGLE of the labels
vertices$id=NA
myleaves=which(is.na( match(vertices$name, edge$from) ))
nleaves=length(myleaves)
vertices$id[ myleaves ] = seq(1:nleaves)
vertices$angle= 90 - 360 * vertices$id / nleaves

# calculate the alignment of labels: right or left
# If I am on the left part of the plot, my labels have currently an angle < -90
vertices$hjust<-ifelse( vertices$angle < -90, 1, 0)

# flip angle BY to make them readable
vertices$angle<-ifelse(vertices$angle < -90, vertices$angle+180, vertices$angle)

# Create a graph object
mygraph <- graph_from_data_frame( edge, vertices=vertices )

# Make the plot
ggraph(mygraph, layout = 'dendrogram', circular = TRUE) + 
  geom_edge_diagonal(colour="grey40") +
  scale_edge_colour_distiller(palette = "RdPu") +
  #geom_node_point(aes(filter = leaf, x = x*1.07, y=y*1.07)) +
  theme_void() +
  theme(
    legend.position="none",
    plot.margin=unit(c(0,0,0,0),"cm"),
    plot.title = element_text(hjust = 0.5,
                              margin = margin(t = 20, b = 20, l = 0, r = 0))  # 标题上方和下方的边距  # 居中标题
  ) +
  labs(title = "Wiki") +  # 添加标题
  coord_fixed(ratio = 1)+  # 设置 x 轴和 y 轴比例为 1:1
  expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))