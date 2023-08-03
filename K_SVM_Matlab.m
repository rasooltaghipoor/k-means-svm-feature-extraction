close all
clc
clear

filename = 'wdbc.data';
delimiterIn = ',';
wdbcData = importdata(filename,delimiterIn);

allData = wdbcData.data;
labels = wdbcData.textdata(:,2);
lbls = ones(569,1);
for i=1:length(labels)
    if(ismember(labels(i) , 'M'))
        lbls(i,1) = 1;
    else
        lbls(i,1) = -1;
    end
end

actual_min = min(allData(:));
actual_max = max(allData(:));
desired_min = -1;
desired_max =  1;
normData = (allData - actual_min)*((desired_max - desired_min)/(actual_max - actual_min)) + desired_min;

Mdata = normData(lbls==1,:);
Bdata = normData(lbls==-1,:);

[M_IDX,M_C] = kmeans(Mdata,3);
[B_IDX,B_C] = kmeans(Bdata,3);
Centers = [M_C;B_C];

F = 30;
for c=1:6
    if(c <= 3)
        ind=(M_IDX==c);
        xc = Mdata(ind,:);
    else
        ind=(B_IDX==(c-3));
        xc = Bdata(ind,:);
    end
    d =abs( 1 - xc(:,1));
    dd = max(d);
    
    for i=1:length(normData)
        for j=1:F
            if(normData(i,j) >= min(xc(:,j)) && normData(i,j) <= max(xc(:,j)))
                fc(j) = 1 - (abs(Centers(c,j) - normData(i,j))/max(abs(Centers(c,j) - xc(:,j))));
            else
                fc(j) = 0;
            end
        end
        featureData(i , c) = mean(fc);
    end    
end

k=10;
N=length(featureData);
N1=length(lbls);
indices = crossvalind('Kfold', N, k);
for i=1:k
    test=(indices==i);
    train=~test;
    datatrain=featureData(train,:);
    ltrain=lbls(train,:);	
    datatest=featureData(test,:);
    ltest=lbls(test,:);
  
    c=10;
    sig=1;
    for j=1:5        
        c=c*10;
        for m=1:5
            sig=sig*10;
            svmStruct = svmtrain(datatrain,ltrain,'boxconstraint',c,'kernel_function','rbf','rbf_sigma',sig);
            lb=svmclassify(svmStruct,datatest);
            prf(j,m)=(length(find(ltest==lb))/length(lb))*100;
        end
    end
   % disp(prf);
    perfrbf(i)=max(max(prf));
end
perf=mean(perfrbf);
disp(perf);



