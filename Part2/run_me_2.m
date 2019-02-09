%% Clear workspace
clc;
clear;
close all;
warning off;
%% Read Train images part(a)
thispath = pwd();

File = fopen([thispath '\faces\train.txt'],'r');
count_tr = 0;
while 1 
    tline = fgetl(File);
   if ~ischar(tline)
       break;
   end
   tline = strsplit(tline);
   im = imread(tline{1});
   count_tr = count_tr + 1;
   X(count_tr,:) = im(:)';
   Y(count_tr) = str2double(tline{2});
end
fclose(File);

%% Read Test images part(a)
File = fopen([thispath '\faces\test.txt'],'r');
count_ts = 0;
while 1 
    tline = fgetl(File);
   if ~ischar(tline)
       break;
   end
   tline = strsplit(tline);
   im = imread(tline{1});
   count_ts = count_ts + 1;
   X_Test(count_ts,:) = im(:)';
   Y_Test(count_ts) = str2double(tline{2});
end
fclose(File);

%% Show random image part(b)
figure;
Rnd_Idx = randi(count_tr);
subplot(1,2,1);
imshow(reshape(X(Rnd_Idx,:),50,50));
title('Random Train Image');
subplot(1,2,2);
Rnd_Idx = randi(count_ts);
imshow(reshape(X_Test(Rnd_Idx,:),50,50));
title('Random Test Image');
pause(0.1);

%% Show mean image of train and test data part(je)
figure;
subplot(1,2,1);
X_tr_Mean = uint8(mean(X));
imshow(reshape(X_tr_Mean,50,50));
title('Mean Train Image');
subplot(1,2,2);
X_ts_Mean = uint8(mean(X_Test));
imshow(reshape(X_ts_Mean,50,50));
title('Mean Test Image');
pause(0.1);

%% normalize data part(d)
X_Train_Sub_Mean = bsxfun(@minus,X,X_tr_Mean);
X_Test_Sub_Mean = bsxfun(@minus,X_Test,X_tr_Mean);

%% Show random image from normaize image part(d)
figure;
Rnd_Idx = randi(count_tr);
subplot(2,2,1);
imshow(reshape(X(Rnd_Idx,:),50,50));
title('Random Train Image');
subplot(2,2,3);
imshow(reshape(X_Train_Sub_Mean(Rnd_Idx,:),50,50));
title('Train - Mean Image');

Rnd_Idx = randi(count_ts);
subplot(2,2,2);
imshow(reshape(X_Test(Rnd_Idx,:),50,50));
title('Random Test Image');
subplot(2,2,4);
imshow(reshape(X_Test_Sub_Mean(Rnd_Idx,:),50,50));
title('Test - Mean Image');
pause(0.1);

%% make SVD and show 10 first image from V part(ho)
A = double(X_Train_Sub_Mean);
[U,S,V] = svd(A);

figure;
for i = 1 : 10
    subplot(2,5,i);
    imshow(reshape(V(:,i)',50,50),[]);
    title([num2str(i) 'th Eigen Face Image']);
end
pause(0.1);

%% reconstruct data using SVD   part(v)
for r = 1 : 200
    Xr = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
    error(r) = mse(A,Xr);
end
figure;
plot(1:r,error,'--','LineWidth',2);
title('MSE error of reconstruction SVD');
xlabel('r');
ylabel('MSE');
legend('MSE Line');
pause(0.1);

%% Face recognition with Logistic Regression part(z)
Acc_arr_Tr = [];
Acc_arr = [];
for r = 1 :200
    fprintf('Face recognition with %d eigen face\n',r);
    F = Dimen_Reducntion_Func(double(X_Train_Sub_Mean),V',r);
    F_Test = Dimen_Reducntion_Func(double(X_Test_Sub_Mean),V',r);
    [B,dev,stats] = mnrfit(F,Y','model','hierarchical');
    
    
    P = mnrval(B,F,'model','hierarchical');
    
    
    [~, Y_Test_Pred] = max(P,[],2);
    Acc_arr_Tr(r) = mean(Y==Y_Test_Pred')*100;
    
    % Test on test images
    P = mnrval(B,F_Test,'model','hierarchical'); 
    
    
    [~, Y_Test_Pred] = max(P,[],2);
    Acc_arr(r) = mean(Y_Test==Y_Test_Pred')*100;
end

figure;
plot(1:r,Acc_arr,'r-o','LineWidth',1);
hold on;
plot(1:r,Acc_arr_Tr,'b--o','LineWidth',1);
title('Accuracy of Logistic Regression  classification');
xlabel('r');
ylabel('Accuracy');
legend('Accuracy Test','Accuracy Train');
pause(0.1);
