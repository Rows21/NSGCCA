combinations = [
    100, 30, 5;
    100, 50, 5;
    100, 100, 5;
    100, 200, 5;
    200, 100, 5;
    400, 100, 5;
    100, 100, 10;
    100, 100, 20
];

% Loop over a single mode
modes = [1,2];
for modeIndex = 1:length(modes)
    mode = modes(modeIndex);
    
    % Iterate over each set of parameters
    for i = 1:size(combinations, 1)
        params = combinations(i, :);
        fprintf('Processing parameters: N=%d, P=%d, S=%d\n', params);

        root = 'E:/res';
        N = params(1);
        Pp = params(2);
        S = params(3);

        if mode == 1
            folder = 'Linear/';
        else
            folder = 'Nonlinear/';
        end
        
        data_path = fullfile(root, 'SNGCCA', 'SNGCCA', 'Data', folder, sprintf('%d_%d_%d/', N, Pp, S));
        
        u1 = zeros(Pp, 0);
        u2 = zeros(Pp, 0); 
        u3 = zeros(Pp, 0); 
        t = table;
        
        for rep = 0:99  
            fprintf('REP=%d\n', rep);
            
            A1 = readmatrix(fullfile(data_path, sprintf('data1_%d.csv', rep)));
            A2 = readmatrix(fullfile(data_path, sprintf('data2_%d.csv', rep)));
            A3 = readmatrix(fullfile(data_path, sprintf('data3_%d.csv', rep)));
            
            % table -> double
            %A1 = table2array(train_data_0);  
            %A2 = table2array(train_data_1);  
            %A3 = table2array(train_data_2);  
            A1 = A1 - mean(A1, 1);
            A2 = A2 - mean(A2, 1);
            A3 = A3 - mean(A3, 1);
            view = {A1, A2, A3};

            % cv
            if rep == 0
                [b0, obj_validate] = cv(view, 5, 1e-2);
            end
            %train_data = load('syntrain_data.mat');
            %train_data = {train_data{1}, train_data{2},train_data{3}};
            %test_data = load('syntest_data.mat');
            %test_data = {test_data.view0, test_data.view1,test_data.view2};
            size_train_data = {size(view{1}) ,size(view{2}),size(view{3})};
            % s =1;
            samp = {size_train_data{1}(1),size_train_data{2}(1),size_train_data{3}(1)};
            fea = {size_train_data{1}(2),size_train_data{2}(2),size_train_data{3}(2)};
            tic;
            %% initailization:
            I = 3; %views
            K = 1; %number of canonical variables
            P = {randn(fea{1},K), randn(fea{2},K), randn(fea{3},K)};
            for j=1:I
            %     [L,MM] = size(X{i});
            %     G{i}=train_data{i}*P{i};
            %     U = randn(samp{1},s);
                G{j}=randn(samp{j},K); % random initialization
            %     [G{i},~,~] = svd(G{i},'econ');
                U{j}=sprandn(samp{j},K,1e-4); %sparsity_level = 1e-4
            %     Q{i}=sprandn(M,K,sparsity_level); 
            end
            %% PDD parallel
            out_iter=100;
            rho=0.2;
            in_iter = 5; % select in_iter=1 for ADMM
            % disp('operating parallel PDD with CG iteration');
            % [U_final_pdd_par,G_pdd,Q_pdd ] = fast_PDD_noreg_par2(train_data,P,I,U,G,out_iter,in_iter,K,rho );
            %% parallel PDD L1
            lamda=b0;
            %disp('running parallel PDD-l1');
            [~,G_pddl1,Q_pddl1 ] = fast_PDD_3L1(view,P,I,U,G,out_iter,in_iter,rho,lamda);
            
            % disp('running parallel PDD-l21');
            % [U_pddl21,G_pddl21,Q_pddl21 ] = fast_PDD_3L21( train_data,P,I,U,G,out_iter,in_iter,rho,lamda);
            % disp(['cost: ',num2str(costXQ(I,X,Q_pddl21))]);
            elapsedTime = toc;
            %disp(['ÔËÐÐÊ±¼ä£º' num2str(elapsedTime) ' Ãë']);
            % save 
            u1(:,rep+1) = Q_pddl1{1}; 
            u2(:,rep+1) = Q_pddl1{2}; 
            u3(:,rep+1) = Q_pddl1{3}; 
            t = [t;table(elapsedTime)];
        
        end
        
        path = ['E:/res/SNGCCA/SNGCCA/Simulation/', folder, '/', num2str(N), '_', num2str(Pp), '_', num2str(S), '/'];
        writematrix(u1',[path 'pdd_u1.csv']);
        writematrix(u2',[path 'pdd_u2.csv']);
        writematrix(u3',[path 'pdd_u3.csv']);
        writetable(t,[path 'pdd_t.csv']);

    end
end


