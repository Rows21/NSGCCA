% Read realdata
root = 'E:\res\SNGCCA\SNGCCA\RealData\newData';
filename = fullfile(root, 'mRNA_expression_standardized.xlsx');
dataTable = readmatrix(filename);
A1 = dataTable(:, 2:end)';
[r1,c1] = size(A1);
u1 = zeros(c1, 0);
filename = fullfile(root, 'DNA_methylation_standardized.xlsx');
dataTable = readmatrix(filename);
A2 = dataTable(:, 2:end)';
[r2,c2] = size(A2);
u2 = zeros(c2, 0);
filename = fullfile(root, 'microRNA_expression_standardized.xlsx');
dataTable = readmatrix(filename);
A3 = dataTable(:, 2:end)';
[r3,c3] = size(A3);
u3 = zeros(c3, 0);

train_data = {A1, A2, A3};
[b0, obj_validate] = cv(train_data, 5, 1e-2);
size_train_data = {size(train_data{1}) ,size(train_data{2}),size(train_data{3})};
% s =1;
samp = {size_train_data{1}(1),size_train_data{2}(1),size_train_data{3}(1)};
fea = {size_train_data{1}(2),size_train_data{2}(2),size_train_data{3}(2)};
tic;
%% initailization:
I = 3; %views
K = 1; %number of canonical variables
P = {randn(fea{1},K), randn(fea{2},K), randn(fea{3},K)};
for j=1:I
    G{j}=randn(samp{j},K); % random initialization
    %     [G{i},~,~] = svd(G{i},'econ');
    U{j}=sprandn(samp{j},K,1e-4); %sparsity_level = 1e-4
    %     Q{i}=sprandn(M,K,sparsity_level); 
end
%% PDD parallel
out_iter=100;
rho=0.2;
in_iter = 4; % select in_iter=1 for ADMM
% disp('operating parallel PDD with CG iteration');
% [U_final_pdd_par,G_pdd,Q_pdd ] = fast_PDD_noreg_par2(train_data,P,I,U,G,out_iter,in_iter,K,rho );
%% parallel PDD L1
lamda=b0;
%disp('running parallel PDD-l1');
[~,G_pddl1,Q_pddl1 ] = fast_PDD_3L1(train_data,P,I,U,G,out_iter,in_iter,rho,lamda);
            
% disp('running parallel PDD-l21');
% [U_pddl21,G_pddl21,Q_pddl21 ] = fast_PDD_3L21( train_data,P,I,U,G,out_iter,in_iter,rho,lamda);
% disp(['cost: ',num2str(costXQ(I,X,Q_pddl21))]);
elapsedTime = toc;
%disp(['ÔËÐÐÊ±¼ä£º' num2str(elapsedTime) ' Ãë']);
% save 
u1(:,1) = Q_pddl1{1}; 
u2(:,1) = Q_pddl1{2}; 
u3(:,1) = Q_pddl1{3}; 
%t = [t;table(elapsedTime)];
        
path = ['E:/res/SNGCCA/SNGCCA/Realdata/', 'respdd', '/'];
writematrix(u1',[path 'pdd_u1.csv']);
writematrix(u2',[path 'pdd_u2.csv']);
writematrix(u3',[path 'pdd_u3.csv']);
writetable(t,[path 'pdd_t.csv']);
