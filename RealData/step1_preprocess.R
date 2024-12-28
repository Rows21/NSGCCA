library('data.table')
library('openxlsx')

#obtained from https://www.cell.com/cms/10.1016/j.ccell.2018.03.014/attachment/ca04da67-7e4b-4930-8eb8-5c941eebf6b7/mmc4.xlsx
#delete the first row
subtype=read.xlsx("mmc4.xlsx", sheet=1) 

#obtained from http://firebrowse.org/?cohort=BRCA&download_dialog=true#
#clinical=as.matrix(fread('BRCA.clin.merged.txt',header=F))
#row.names(clinical)=clinical[,1]
#clinical=clinical[,-1]
  
Exp.ori=as.matrix(fread('BRCA.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt',header=T))
Exp.ori=Exp.ori[-1,]
row.names(Exp.ori)=Exp.ori[,1]
Exp.ori=Exp.ori[,-1]
class(Exp.ori)='numeric'

##gdac.broadinstitute.org_BRCA.Merge_methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.Level_3.2016012800.0.0.tar.gz
##is obtained from http://firebrowse.org/?cohort=BRCA&download_dialog=true#
#ME=fread('BRCA.methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt',header=T,nrows=2)
#col=c(1:5,seq(6,dim(ME)[2],by=4))
#fwrite(ME[,..col], file = "TCGA_BRCA_humanmethylation450.txt",sep=" ")
#Meth.ori=fread('TCGA_BRCA_humanmethylation450.txt',header=T,nrows=2)
##only can obtain 721 shared samples

#So, we obtain the DNA methylation data from https://gdc.cancer.gov/about-data/publications/brca_2012
Meth.ori=as.matrix(read.table('BRCA.methylation.27k.450k.txt',header=T))

dim(Exp.ori) #20531  1212
dim(Meth.ori)#21986   940

colnames(Exp.ori)[1]
colnames(Meth.ori)[2]
subtype[1,1]


Exp.id=substr(colnames(Exp.ori), 6,15)
Exp.id2=substr(Exp.id,9,10)
not01=which(Exp.id2!='01')#'01': Primary Solid Tumor
Exp.id[not01]='NA'
Exp.id=substr(Exp.id, 1,7)


Meth.id=substr(colnames(Meth.ori), 6,15)
Meth.id=gsub("[[:punct:]]", "-", Meth.id)
Meth.id2=substr(Meth.id,9,10)
not01=which(Meth.id2!='01')
Meth.id[not01]='NA'
Meth.id=substr(Meth.id, 1,7)

subtype.id=substr(subtype$Sample.ID, 6,15)
subtype.id.PAM50=subtype.id[subtype$BRCA_Subtype_PAM50%in%c("Basal","LumA","LumB","Her2","Normal")]

com.id=intersect(intersect(Exp.id,Meth.id),subtype.id.PAM50)
length(com.id) #798
length(unique(com.id))#798
sum(Exp.id%in%com.id)#798
sum(Meth.id%in%com.id)#798
sum(subtype.id%in%com.id)#798

###miRNA
##BRCA.mirnaseq__illuminahiseq_mirnaseq__bcgsc_ca__Level_3__miR_gene_expression__data.data.txt is obtained from http://firebrowse.org/?cohort=BRCA&download_dialog=true#
#miRNA.ori=as.matrix(fread('BRCA.mirnaseq__illuminahiseq_mirnaseq__bcgsc_ca__Level_3__miR_gene_expression__data.data.txt',header=T))
#col=seq(2,dim(miRNA.ori)[2],by=3)
#row.names(miRNA.ori)=miRNA.ori[,1]
#miRNA.ori=miRNA.ori[,col]
#colnames(miRNA.ori)[1]
#miRNA.id=substr(colnames(miRNA.ori), 6,12)
#sum(miRNA.id%in%com.id)# 553
##only can obtain 553 shared samples

#So, we obtain the miRNA data from https://gdc.cancer.gov/about-data/publications/brca_2012
miRNA.ori=as.matrix(fread('BRCA.780.precursor.txt'))
row.names(miRNA.ori)=miRNA.ori[,1]
miRNA.ori=miRNA.ori[,-1]
class(miRNA.ori)='numeric'
colnames(miRNA.ori)[1]
miRNA.id=substr(colnames(miRNA.ori), 6,15)
miRNA.id2=substr(miRNA.id,9,10)
not01=which(miRNA.id2!='01')
miRNA.id[not01]='NA'
miRNA.id=substr(miRNA.id, 1,7)
sum(miRNA.id%in%com.id)# 683

com.id=intersect(miRNA.id,com.id)
length(com.id)#683
length(unique(com.id))#683 participants

#find the common set of subjects
Exp.com=Exp.ori[,match(com.id,Exp.id,nomatch=0)]
Meth.com=Meth.ori[,match(com.id,Meth.id,nomatch=0)]
miRNA.com=miRNA.ori[,match(com.id,miRNA.id,nomatch=0)]
subtype.com=subtype[match(com.id,subtype.id,nomatch=0),]

dim(Exp.com)#20531   683
dim(Meth.com)#21986   683
dim(miRNA.com)#1046   683
dim(subtype.com)#683   28


label=numeric(dim(subtype.com)[1])
label[which(subtype.com$BRCA_Subtype_PAM50=='Basal')]=1
label[which(subtype.com$BRCA_Subtype_PAM50=='LumA')]=2
label[which(subtype.com$BRCA_Subtype_PAM50=='LumB')]=3
label[which(subtype.com$BRCA_Subtype_PAM50=='Her2')]=4
label[which(subtype.com$BRCA_Subtype_PAM50=='Normal')]=5

sum(label==1)#111
sum(label==2)#346
sum(label==3)#151
sum(label==4)#56
sum(label==5)#19

Exp.com=Exp.com[,label!=5]
Meth.com=Meth.com[,label!=5]
miRNA.com=miRNA.com[,label!=5]
subtype.com=subtype.com[label!=5,]
label=label[label!=5]

write.table(label,file='PAM50label664.txt',row.names = FALSE,col.names = FALSE)
write.table(subtype.com,file='subtype664.txt',row.names = FALSE,col.names = TRUE)
######
Exp.com=log2(Exp.com)
Exp.com[Exp.com==-Inf]=NA

Meth.mat=sqrt(Meth.com) ##take square root of methylation data

miRNA.com=log2(miRNA.com)
miRNA.com[miRNA.com==-Inf]=NA

#check missing values
Exp.na=apply(Exp.com,1,function(x) sum(is.na(x))/dim(Exp.com)[2])
max(Exp.na)#1
Exp.mat=Exp.com[-which(Exp.na>0.5),]
dim(Exp.mat)#17613   664

Meth.na=apply(Meth.mat,1,function(x) sum(is.na(x))/dim(Meth.mat)[2])
max(Meth.na)# 0.06927711

miRNA.na=apply(miRNA.com,1,function(x) sum(is.na(x))/dim(miRNA.com)[2])
max(miRNA.na)#1
miRNA.mat=miRNA.com[-which(miRNA.na>0.5),]
dim(miRNA.mat)#437 664

#impute missing values
#install.packages("BiocManager")
#BiocManager::install("impute")
library(impute)#version 1.58.0
Exp.imputed=impute.knn(Exp.mat,rng.seed=0)$data 
Meth.imputed=impute.knn(Meth.mat,rng.seed=0)$data 
miRNA.imputed=impute.knn(miRNA.mat,rng.seed=0)$data 


# select the most variably expressed genes/methylated probes
Exp.sd=apply(Exp.imputed,1,sd)
Meth.sd=apply(Meth.imputed,1,sd)

Exp.select = Exp.imputed[Exp.sd>quantile(Exp.sd,0.85),]
Meth.select = Meth.imputed[Meth.sd>quantile(Meth.sd,0.85),]

dim(Exp.select)#2642  664
dim(Meth.select)#3298  664
dim(miRNA.imputed)#437 664

#centering
Exp.select.centered=t(scale(t(Exp.select), center = T, scale =F))
Meth.select.centered=t(scale(t(Meth.select), center = T, scale =F))
miRNA.select.centered=t(scale(t(miRNA.imputed), center = T, scale =F))

               
write.table(Exp.select.centered,file='Exp664.txt',row.names = FALSE,col.names = FALSE)
write.table(Meth.select.centered,file='Meth664.txt',row.names = FALSE,col.names = FALSE)
write.table(miRNA.select.centered,file='miRNA664.txt',row.names = FALSE,col.names = FALSE)


write.table(rownames(Exp.select.centered),file='Exp664_genes.txt',row.names = FALSE,col.names = FALSE)
write.table(rownames(Meth.select.centered),file='Meth664_probes.txt',row.names = FALSE,col.names = FALSE)
write.table(rownames(miRNA.select.centered),file='miRNA664_miRNA.txt',row.names = FALSE,col.names = FALSE)

