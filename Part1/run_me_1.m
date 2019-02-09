%% Clear workspace
clc;
clear;
close all;
randn('seed',10);

%% Make data
mu1=[10 10]';
mu2=[22 10]';
sig1 = [4,4;4,9];
sig2 = sig1 ;
N = 1000;
x1 = mvnrnd(mu1,sig1,N);
x2=mvnrnd(mu2,sig2,N);
x = [x1;x2];

%% Normalize
x = x - mean(x);

%% Do PCA
[coeff,score,latent,tsquared,explained,mu] = pca(x,'Centered',false);

%% Find PCA line
m = coeff(2)/coeff(1);
fun=@(x)m*x;

%% Project data on 
pc_vector = coeff(:,1);
x_projection = score(:,1);
x_projection1=(x_projection(1:N)./(pc_vector'*pc_vector))       *pc_vector';
x_projection2=(x_projection(N+1:2*N)./(pc_vector'*pc_vector))*pc_vector';


%% Plot PCA line    (part a)
figure;
scatter(x(1:N,1), x(1:N,2));
hold on;
scatter(x(N+1:end,1), x(N+1:end,2));
fplot(fun,[min(x(:,1))-2,max(x(:,1))+2],'LineWidth',2);
title('PCA Line');
legend('Class1','Class2','PCA Line');
ylim([min(x(:,2))-2,max(x(:,2))+2]);
xlim([min(x(:,1))-2,max(x(:,1))+2]);

%% Plot projected data    (part b)
figure;
scatter(x(1:N,1), x(1:N,2));
hold on;
scatter(x(N+1:end,1), x(N+1:end,2));

scatter(x_projection1(:,1), x_projection1(:,2));
scatter(x_projection2(:,1), x_projection2(:,2));
fplot(fun,[min(x(:,1))-2,max(x(:,1))+2],'LineWidth',2);
title('PCA Projection');
legend('Class1','Class2','Projected Class1','Projected Class2','PCA Line');
ylim([min(x(:,2))-2,max(x(:,2))+2]);
xlim([min(x(:,1))-2,max(x(:,1))+2]);

%% Reconstruct data using PCA part(d)
rec = (score)*coeff';
Mse_Error = mse(rec,x);
figure;
scatter(x(1:N,1), x(1:N,2),'bo','linewidth',2);
hold on;
scatter(x(N+1:end,1), x(N+1:end,2),'ro','linewidth',2);
scatter(rec(1:N,1), rec(1:N,2),'y*');
scatter(rec(N+1:end,1), rec(N+1:end,2),'g*');
legend('Class1','Class2','Reconstruct Class1','Reconstruct Class2');
title(['Reconstruct data with MSE error = ' num2str(Mse_Error)]);
