library(TCGAbiolinks) # version 2.34.0 
library(DESeq2) # version 1.46.0
library(tidyverse) # version 2.0.0
library(sesameData) # version 1.24.0
library(sesame) # version 1.24.0
library(impute) # version 1.80.0
library(sva) # version 3.54.0
library(edgeR) # version 4.4.1
library(openxlsx)
# The latest time to download data is 1/16-17/2025

#### mRNA expression ####
# Genome of reference: hg38
mRNA.query <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  experimental.strategy = "RNA-Seq",
  sample.type = c("Primary Tumor")
)

GDCdownload(query = mRNA.query)

mRNA.dataPrep <- GDCprepare(query = mRNA.query)
save(mRNA.dataPrep, file = "mRNA.dataPrep.rda")

table(colData(mRNA.dataPrep)$gender)
#female   male 
#1098     12

# only consider females
female.id <- colData(mRNA.dataPrep)$gender=="female"
female.id[is.na(female.id)] <- FALSE

# filter variables
mRNA.counts_matrix <- assays(mRNA.dataPrep)$unstranded[,female.id]
dim(mRNA.counts_matrix) # 60660  1098

keep <- filterByExpr(mRNA.counts_matrix)
sum(keep)

mRNA.counts_matrix.filtered <- mRNA.counts_matrix[keep,]
dim(mRNA.counts_matrix.filtered) # 18213  1098

# normalization by DESeq2
dds <- DESeqDataSetFromMatrix(mRNA.counts_matrix.filtered, colData = colData(mRNA.dataPrep)[female.id,], design = ~1)
dds2 <- DESeq2::estimateSizeFactors(dds)
mRNA.dataNorm <- DESeq2::counts(dds2, normalized = TRUE) 

sum(is.na(mRNA.dataNorm))

log2.mRNA.dataNorm <- log2(mRNA.dataNorm+1)

dim(log2.mRNA.dataNorm) # 18213  1098


#### microRNA expression ####
# Genome of reference: hg38
microRNA.query <- GDCquery(
  project = "TCGA-BRCA", 
  data.category = "Transcriptome Profiling", 
  data.type = "miRNA Expression Quantification",
  experimental.strategy = "miRNA-Seq",
  sample.type = c("Primary Tumor")
)

GDCdownload(query = microRNA.query, method="api", files.per.chunk =548)

microRNA.dataPrep <- GDCprepare(query = microRNA.query)
save(microRNA.dataPrep, file = "microRNA.dataPrep.rda")

rownames(microRNA.dataPrep) <- microRNA.dataPrep$miRNA_ID

# using read_count's data 
read_countData <-  colnames(microRNA.dataPrep)[grep("count", colnames(microRNA.dataPrep))]
microRNA.counts_matrix <- microRNA.dataPrep[,read_countData]
colnames(microRNA.counts_matrix) <- gsub("read_count_","", read_countData)

dim(microRNA.counts_matrix) # 1881 1096

female.id2 <- str_sub(colnames(microRNA.counts_matrix),1,12)%in%colData(mRNA.dataPrep)$patient[female.id]

microRNA.counts_matrix <- microRNA.counts_matrix[,female.id2]

dim(microRNA.counts_matrix) # 1881 1081

microRNA.colData <- data.frame(condition=numeric(sum(female.id2)), row.names=colnames(microRNA.counts_matrix))

# normalization by DESeq2
dds <- DESeqDataSetFromMatrix(microRNA.counts_matrix, colData = microRNA.colData, design = ~1)
dds2 <- DESeq2::estimateSizeFactors(dds)
microRNA.dataNorm <- DESeq2::counts(dds2, normalized = TRUE) 

sum(is.na(microRNA.dataNorm))

log2.microRNA.dataNorm <- log2(microRNA.dataNorm+1)

dim(log2.microRNA.dataNorm) #1881 1081

#### DNA methylation ####
# Genome of reference: hg38
####* Illumina methylation 450k #### 
DNA450.query <- GDCquery(
  project = "TCGA-BRCA", 
  data.category = "DNA Methylation", 
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 450",
  sample.type = c("Primary Tumor")
)

GDCdownload(query = DNA450.query, files.per.chunk = 40)

DNA450.dataPrep <- GDCprepare(query = DNA450.query)
save(DNA450.dataPrep, file = "DNA450.dataPrep.rda")

DNA450.beta <- assay(DNA450.dataPrep)


####* Illumina methylation 27k #### 
DNA27.query <- GDCquery(
  project = "TCGA-BRCA", 
  data.category = "DNA Methylation", 
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 27",
  sample.type = c("Primary Tumor")
)

GDCdownload(query = DNA27.query)

DNA27.dataPrep <- GDCprepare(query = DNA27.query)
save(DNA27.dataPrep, file = "DNA27.dataPrep.rda")

DNA27.beta <- assay(DNA27.dataPrep)

####* combine 450k and 27k ####
sum(is.na(DNA450.beta)) # 59408530
dim(DNA450.beta) #485577    793

sum(is.na(DNA27.beta)) # 886606
dim(DNA27.beta) #27578   314

common_cpg <- intersect(rownames(DNA450.beta), rownames(DNA27.beta))
DNA450.beta.common <- DNA450.beta[common_cpg, ]
DNA27.beta.common <- DNA27.beta[common_cpg, ]

dim(DNA450.beta.common) # 25978   793
dim(DNA27.beta.common) # 25978   314

intersect(str_sub(colnames(DNA450.beta.common),1,19),
          str_sub(colnames(DNA27.beta.common),1,19))
#no common sample portion


female.id.450 <- str_sub(colnames(DNA450.beta.common),1,12)%in%colData(mRNA.dataPrep)$patient[female.id]
n.female.id.450 <- sum(female.id.450) #  781

female.id.27 <- str_sub(colnames(DNA27.beta.common),1,12)%in%colData(mRNA.dataPrep)$patient[female.id]
n.female.id.27 <- sum(female.id.27) #  310

DNA.beta <- cbind(DNA450.beta.common[,female.id.450],DNA27.beta.common[,female.id.27])
dim(DNA.beta) # 25978  1091

##remove cpg sites with missing rate > 10%
DNA.beta.filtered <- DNA.beta[rowMeans(is.na(DNA.beta)) <= 0.1, ]
dim(DNA.beta.filtered) # 23495  1091

##impute
DNA.beta.imputed <- impute.knn(DNA.beta.filtered)$data
max(DNA.beta.imputed)
min(DNA.beta.imputed)

##Batch Correction
batch <- c(rep("450k", n.female.id.450),rep("27k", n.female.id.27))

# convert to M value
DNA.Mvalue.imputed <- log2((DNA.beta.imputed + 0.001)/(1 - DNA.beta.imputed + 0.001))

DNA.Mvalue.corrected <- ComBat(dat = DNA.Mvalue.imputed, batch = batch, par.prior = TRUE)

dim(DNA.Mvalue.corrected) # 23495  1091

# check batch effect again
get_p_value <- function(row_data, batch) {
  model <- lm(row_data ~ batch)
  summary(model)$coefficients[2, 4]  # Extract the p-value for the group variable
}

p_values <- map_dbl(as.data.frame(t(DNA.Mvalue.corrected)), get_p_value, batch)
sum(p_values < 0.05) # 0

#### Find the common set of objects  #####
#log2.mRNA.dataNorm
#log2.microRNA.dataNorm
#DNA.Mvalue.corrected

obj.mRNA <- str_sub(colnames(log2.mRNA.dataNorm),1,19)
obj.microRNA <- str_sub(colnames(log2.microRNA.dataNorm),1,19)
obj.DNA <- str_sub(colnames(DNA.Mvalue.corrected),1,19)

length(obj.mRNA) #1098
length(obj.microRNA)#1081
length(obj.DNA) #1091

length(unique(obj.mRNA)) #1093
length(unique(obj.microRNA))#1076
length(unique(obj.DNA)) #1089

obj.com <- intersect(obj.mRNA,obj.microRNA)
obj.com <- intersect(obj.DNA,obj.com)

length(obj.com) #1066

sum(obj.mRNA%in%obj.com) #1071
sum(obj.microRNA%in%obj.com) #1071
sum(obj.DNA%in%obj.com) #1068

log2.mRNA.com <- log2.mRNA.dataNorm[,obj.mRNA%in%obj.com]
log2.microRNA.com <- log2.microRNA.dataNorm[,obj.microRNA%in%obj.com]
DNA.Mvalue.com <- DNA.Mvalue.corrected[,obj.DNA%in%obj.com]

dim(log2.mRNA.com) #18213  1071
dim(log2.microRNA.com) #1881 1071
dim(DNA.Mvalue.com) #23495  1068

patient.id <- unique(str_sub(obj.com, 1, 12)) 

length(obj.com) # 1066
length(patient.id) # 1059

clinical <- colData(mRNA.dataPrep)
dim(clinical) #1111   88

clinical <- clinical[str_sub(clinical$barcode,1,19)%in%obj.com,]
dim(clinical) #1071   88

clinical <- clinical[clinical$is_ffpe == F,]
dim(clinical) #1064   88

length(unique(clinical$patient)) # 1059

table(clinical$paper_BRCA_Subtype_PAM50,useNA="always")
#Basal   Her2   LumA   LumB Normal   <NA> 
#  185     81    556    202     40      0

clinical$patient[duplicated(clinical$patient)]
clinical$sample[duplicated(clinical$sample)]

sort(clinical$barcode[clinical$patient%in%clinical$patient[duplicated(clinical$patient)]])


log2.mRNA.com <- log2.mRNA.com[,str_sub(colnames(log2.mRNA.com),1,19)%in%str_sub(clinical$barcode,1,19)]
dim(log2.mRNA.com) # 18213  1064

log2.microRNA.com <- log2.microRNA.com[,str_sub(colnames(log2.microRNA.com),1,19)%in%str_sub(clinical$barcode,1,19)]
dim(log2.microRNA.com) # 1881  1064

DNA.Mvalue.com <- DNA.Mvalue.com[,str_sub(colnames(DNA.Mvalue.com),1,19)%in%str_sub(clinical$barcode,1,19)]
dim(DNA.Mvalue.com) # 23495  1061

dim(clinical) # 1064   88
length(unique(clinical$patient)) # 1059

# multiple observations for the same patients
sort(colnames(log2.mRNA.com)[str_sub(colnames(log2.mRNA.com),1,12)%in%clinical$patient[duplicated(clinical$patient)]])
sort(colnames(log2.microRNA.com)[str_sub(colnames(log2.microRNA.com),1,12)%in%clinical$patient[duplicated(clinical$patient)]])
sort(colnames(DNA.Mvalue.com)[str_sub(colnames(DNA.Mvalue.com),1,12)%in%clinical$patient[duplicated(clinical$patient)]])


#### remove multiple observations for the same patients
# mRNA
log2.mRNA.com.patient <- str_sub(colnames(log2.mRNA.com),1,12)
log2.mRNA.com.barcode.delete <- NULL

for(id in log2.mRNA.com.patient[duplicated(log2.mRNA.com.patient)]){
  #5 patients have 2 observations
  #select the one obs with smaller number of zeros
  temp <- colSums(log2.mRNA.com[,log2.mRNA.com.patient==id]==0)
  log2.mRNA.com.barcode.delete <- c(log2.mRNA.com.barcode.delete,
                                    attributes(which(temp==max(temp)))$names[1])
}

log2.mRNA.com <- log2.mRNA.com[,!colnames(log2.mRNA.com)%in%log2.mRNA.com.barcode.delete]
dim(log2.mRNA.com) # 18213  1059


# microRNA
log2.microRNA.com.patient <- str_sub(colnames(log2.microRNA.com),1,12)
log2.microRNA.com.barcode.delete <- NULL

for(id in log2.microRNA.com.patient[duplicated(log2.microRNA.com.patient)]){
  #5 patients have 2 observations
  #select the one obs with smaller number of zeros
  temp <- colSums(log2.microRNA.com[,log2.microRNA.com.patient==id]==0)
  log2.microRNA.com.barcode.delete <- c(log2.microRNA.com.barcode.delete,
                                        attributes(which(temp==max(temp)))$names[1])
}

log2.microRNA.com <- log2.microRNA.com[,!colnames(log2.microRNA.com)%in%log2.microRNA.com.barcode.delete]
dim(log2.microRNA.com) # 1881  1059


# DNA
DNA.Mvalue.com.patient <- str_sub(colnames(DNA.Mvalue.com),1,12)
DNA.Mvalue.com.barcode.delete <- NULL

for(id in DNA.Mvalue.com.patient[duplicated(DNA.Mvalue.com.patient)]){
  #still have 2 patients have 2 observations
  #select the one obs with high methylation
  temp <- colSums(DNA.Mvalue.com[,DNA.Mvalue.com.patient==id])
  DNA.Mvalue.com.barcode.delete <- c(DNA.Mvalue.com.barcode.delete,
                                     attributes(which(temp==min(temp)))$names[1])
}

DNA.Mvalue.com <- DNA.Mvalue.com[,!colnames(DNA.Mvalue.com)%in%DNA.Mvalue.com.barcode.delete]
dim(DNA.Mvalue.com) # 23495  1059

#### Clinical data ####
clinical <- clinical[clinical$barcode%in%colnames(log2.mRNA.com),]
dim(clinical) # 1059   88

# select variables in clinical data
colnames(clinical)

selectVar <- c("patient", "vital_status", "days_to_death", "days_to_last_follow_up", 
               "paper_age_at_initial_pathologic_diagnosis", "race", "ethnicity",
               "paper_pathologic_stage", "paper_BRCA_Pathology", "paper_BRCA_Subtype_PAM50")
clinical <- clinical[, selectVar]

table(clinical$paper_BRCA_Subtype_PAM50, useNA="always")
# Basal   Her2   LumA   LumB Normal   <NA> 
#   183     81    553    202     40      0 

clinical.2 <- GDCquery_clinic(project = "TCGA-BRCA", type = "clinical")
save(clinical.2, file = "clinical.2.rda")

clinical.2 <- clinical.2[,c("submitter_id","treatments_pharmaceutical_treatment_or_therapy","treatments_radiation_treatment_or_therapy")]

clinical.data <- left_join(as.data.frame(clinical), clinical.2, by = c("patient" = "submitter_id")) 

dim(clinical.data) # 1059   12

sum(is.na(clinical.data$paper_age_at_initial_pathologic_diagnosis))

table(clinical.data$race, useNA="always")
#american indian or alaska native                            asian 
#                               1                               61 
#       black or african american                     not reported 
#                             179                               86 
#                           white                             <NA> 
#                             732                                0 
clinical.data$race[clinical.data$race%in%c("american indian or alaska native","not reported")] = "other"

table(clinical.data$ethnicity, useNA="always")
table(clinical.data$paper_pathologic_stage, useNA="always")
table(clinical.data$paper_BRCA_Pathology, useNA="always")
table(clinical.data$paper_BRCA_Subtype_PAM50, useNA="always")
table(clinical.data$treatments_pharmaceutical_treatment_or_therapy, useNA="always")
table(clinical.data$treatments_radiation_treatment_or_therapy, useNA="always")


time = ifelse(clinical.data$vital_status=="Alive", 
              clinical.data$days_to_last_follow_up,
              clinical.data$days_to_death)

time[time<0 | is.na(time)] # -7 NA
# remove the data of the two patients
rm.patient.index <- which(time<0 | is.na(time))
rm.patient.id <- clinical.data$patient[rm.patient.index]
rm.patient.id
# "TCGA-PL-A8LV" "TCGA-E9-A245"
# whose races are
# "black or african american" "white"

clinical.data <- clinical.data[-rm.patient.index,]
dim(clinical.data) # 1057   12

mRNA.patient.id <- str_sub(colnames(log2.mRNA.com),1,12)
log2.mRNA.com <- log2.mRNA.com[,!mRNA.patient.id%in%rm.patient.id]
mRNA.patient.id <- str_sub(colnames(log2.mRNA.com),1,12)

Match.mRNA = match(clinical.data$patient, mRNA.patient.id, nomatch=0)
log2.mRNA.com = log2.mRNA.com[, Match.mRNA]


microRNA.patient.id <- str_sub(colnames(log2.microRNA.com),1,12)
log2.microRNA.com <- log2.microRNA.com[,!microRNA.patient.id%in%rm.patient.id]
microRNA.patient.id <- str_sub(colnames(log2.microRNA.com),1,12)

Match.microRNA = match(clinical.data$patient, microRNA.patient.id, nomatch=0)
log2.microRNA.com = log2.microRNA.com[, Match.microRNA]


DNA.patient.id <- str_sub(colnames(DNA.Mvalue.com),1,12)
DNA.Mvalue.com <- DNA.Mvalue.com[,!DNA.patient.id%in%rm.patient.id]
DNA.patient.id <- str_sub(colnames(DNA.Mvalue.com),1,12)

Match.DNA = match(clinical.data$patient, DNA.patient.id, nomatch=0)
DNA.Mvalue.com = DNA.Mvalue.com[, Match.DNA]

sum(str_sub(colnames(log2.mRNA.com),1,12)==clinical.data$patient)
sum(str_sub(colnames(log2.microRNA.com),1,12)==clinical.data$patient)
sum(str_sub(colnames(DNA.Mvalue.com),1,12)==clinical.data$patient)

dim(log2.mRNA.com) #18213  1057
dim(log2.microRNA.com) #1881 1057
dim(DNA.Mvalue.com) #1881 1057
dim(clinical.data) #1057   12

table(clinical.data$paper_BRCA_Subtype_PAM50, useNA="always")
#Basal   Her2   LumA   LumB Normal   <NA> 
#  182     81    552    202     40      0

table(clinical.data$vital_status, useNA="always")
#Alive  Dead  <NA> 
#  911   146     0 

#### Filter genomic variables #### 
# remove mRNA with low variance. Note that filterByExpr is done before normalization.
log2.mRNA.filtered <- log2.mRNA.com[apply(log2.mRNA.com,1,sd)>1.5,] 
dim(log2.mRNA.filtered) # 2596 1057

# remove microRNAs with low reads
log2.microRNA.filtered <- log2.microRNA.com[rowSums(log2.microRNA.com==0)<0.5*ncol(log2.microRNA.com),] 
dim(log2.microRNA.filtered) # 523 1057

# remove cpg sites with extreme low or high mean methylation level (beta < 0.2 or beta > 0.8, equivalently, M < -2 or M > 2)
DNA.Mvalue.filtered <- DNA.Mvalue.com[abs(apply(DNA.Mvalue.com,1,mean)) <= 2,]
dim(DNA.Mvalue.filtered) # 6154 1057

DNA.Mvalue.sd <- apply(DNA.Mvalue.filtered,1,sd)

DNA.Mvalue.filtered <- DNA.Mvalue.filtered[DNA.Mvalue.sd >= median(DNA.Mvalue.sd),] 
dim(DNA.Mvalue.filtered) # 3077 1057


write.xlsx(as.data.frame(log2.mRNA.filtered), "mRNA_expression.xlsx", rowNames = TRUE)
write.xlsx(as.data.frame(log2.microRNA.filtered), "microRNA_expression.xlsx", rowNames = TRUE)
write.xlsx(as.data.frame(DNA.Mvalue.filtered), "DNA_methylation.xlsx", rowNames = TRUE)
write.xlsx(clinical.data, "clinical_data.xlsx")

# standardized data
log2.mRNA.standardized <- t(scale(t(log2.mRNA.filtered), center=T, scale=T))
log2.microRNA.standardized <- t(scale(t(log2.microRNA.filtered), center=T, scale=T))
DNA.Mvalue.standardized <- t(scale(t(DNA.Mvalue.filtered), center=T, scale=T))

write.xlsx(as.data.frame(log2.mRNA.standardized), "mRNA_expression_standardized.xlsx", rowNames = TRUE)
write.xlsx(as.data.frame(log2.microRNA.standardized), "microRNA_expression_standardized.xlsx", rowNames = TRUE)
write.xlsx(as.data.frame(DNA.Mvalue.standardized), "DNA_methylation_standardized.xlsx", rowNames = TRUE)

