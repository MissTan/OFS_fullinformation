function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = OFSGD_sample_truncate(X, Y, options, id_list)
NumFeature=options.NumFeature;
[n,d]       = size(X);
ID = id_list;
err_count = 0;
t_tick = options.t_tick;
mistakes = [];
mistakes_idx = [];
SV = [];
SVs = [];
TMs=[];
eta = 0.2;
lambda = 0.01;
k = 2;
w=zeros(size(X,2),1);     % initialize the weight vector
%% loop
tic
X_rec = zeros(1, size(X, 2));
for t = 1:length(ID),
    id = ID(t);

    %% prediction
    %x_t=X(id,:)';
    %%%
    x_t = X(id, :);
    if t >= (length(ID)/2),
        
        x_rec = x_t ~= 0; 
        X_rec = X_rec + x_rec;
        X_rec = truncate(X_rec, 10);
        t = (X_rec ~= 0);
        x_t = x_t .* t;
    end
    x_t = x_t';
    %%%
    f_t=w'*x_t;
    y_t=Y(id);

    if y_t*f_t<=0,
        err_count=err_count+1;
    end 
    if y_t*f_t<=1,

        w= w+eta*y_t*x_t;
        w = w*min(1,1/(sqrt(lambda)*norm(w)));
        w=truncate(w,NumFeature);
        SV = [SV id];
    end

    run_time = toc;
%     if (t==k)
%         k = 2*k;
%         mistakes = [mistakes err_count/t];
%         mistakes_idx = [mistakes_idx t];
%         SVs = [SVs length(SV)];
%         TMs=[TMs run_time];
%     end
    
    if (mod(t, 10) == 0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end

classifier.SV = SV;
classifier.w = w;
run_time = toc;

