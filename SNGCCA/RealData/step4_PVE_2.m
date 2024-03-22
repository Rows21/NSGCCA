%analyze BRCA data 
label=load('PAM50label664.txt');% subtype labels

X_1=load(['Exp664.txt']);%mRNA
X_2=load(['Meth664.txt']);
X_3=load(['miRNA664.txt']);

X_SNG1 = load(['Exp_score.csv']);
%C_1=load(['C_k_3sets_Exp_centered_sig_level0.05_DGCCA.txt']);%mRNA
%C_2=load(['C_k_3sets_miRNA_centered_sig_level0.05_DGCCA.txt']);
%C_3=load(['C_k_3sets_Meth_centered_sig_level0.05_DGCCA.txt']);

%D_1=load(['D_k_3sets_Exp_centered_sig_level0.05_DGCCA.txt']);%mRNA
%D_2=load(['D_k_3sets_miRNA_centered_sig_level0.05_DGCCA.txt']);
%D_3=load(['D_k_3sets_Meth_centered_sig_level0.05_DGCCA.txt']);


%% SWISS
i = 2642;

round(SWISS(X_1(i,:),[],label),3) 
%round(SWISS(C_1(i,:),[],label),3) 
%round(SWISS(D_1(i,:),[],label),3) 
i = 3298;
round(SWISS(X_2(i,:),[],label),3)
%round(SWISS(C_2(i,:),[],label),3)
%round(SWISS(D_2(i,:),[],label),3)
i = 437;
round(SWISS(X_3(i,:),[],label),3)
%round(SWISS(C_3(i,:),[],label),3)
%round(SWISS(D_3(i,:),[],label),3)

%% test SWISS difference in C and D
paramstruct = struct('nsim',10000,'seed',0)
epval1 = zeros(10,2);
epval2 = zeros(10,2);
epval3 = zeros(10,2);
epval4 = zeros(10,2);
epval5 = zeros(10,2);
epval6 = zeros(10,2);
index1 = [85 2042    5  973 2210 1896 1309  924  984 2460];
index2 = [373  82 142  73 349 118 186 350 180 196];
index3 = [523  263 2184 3198 2794 1529 3015 1746  309  123];
index4 = [905 1678 2357 1839 1133 1136  588 2511 1410 1740];
index5 = [242 176 127 414 122 380 226 181 177   7];
index6 = [2876 2093 1702   16  347  931  232 1299 2122 1040];
for i=1:10
[~, epval1(i,:), ~]=SWISS(C_1(index1(i),:),D_1(index1(i),:),label,paramstruct);
[~, epval2(i,:), ~]=SWISS(C_2(index2(i),:),D_2(index2(i),:),label,paramstruct);
[~, epval3(i,:), ~]=SWISS(C_3(index3(i),:),D_3(index3(i),:),label,paramstruct);
[~, epval4(i,:), ~]=SWISS(C_1(index4(i),:),D_1(index4(i),:),label,paramstruct);
[~, epval5(i,:), ~]=SWISS(C_2(index5(i),:),D_2(index5(i),:),label,paramstruct);
[~, epval6(i,:), ~]=SWISS(C_3(index6(i),:),D_3(index6(i),:),label,paramstruct);
end







