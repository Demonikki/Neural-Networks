% PART G - NETWORK TOPOLOGY%
TrainingN = 50000;
TestingN = 10000;

images = loadMNISTImages('train-images.idx3-ubyte');
row1 = zeros (1, TrainingN);
t_row1 = zeros(1, TestingN);
labels = loadMNISTLabels('train-labels.idx1-ubyte');

t = zeros(10,TrainingN);
t_teach = zeros(10,TestingN);
predict = zeros(10,TrainingN);
t_predict = zeros(10,TestingN);

%convert labels to 10x60k
for i = 1:TrainingN
    row1(i)=1;
    t(labels(i)+1,i)=1;
end

for i = 1:TestingN
    t_row1(i) = 1;
    t_teach(labels(i+TrainingN)+1,i)=1;
end

X1 = cat(1,row1,images(:,1:TrainingN));
t_X1 = cat(1,t_row1,images(:,TrainingN+1:TrainingN+TestingN));

%H = number of rows of input
M = size(images,1);
%%% Forward propagation  %%%

%H = number of hidden nodes
H = 400;    %change to 10 or 400 for part g

%C = classes
C = 10;

%Initialize random weights (0-2) for Input to Hidden and Hidden to Output
Wij = rand(M+1,H); 
Wjk = rand(H+1,C);
Wij = Wij - .5;
Wjk = Wjk - .5;

%Initialize gradient matrices
Dj_bp = zeros(H,TrainingN);
Dk_bp = zeros(C,TrainingN);
Dj_na = zeros(H,TrainingN);
Dk_na = zeros(C,TrainingN);

%Preparing for learning
learnRate = 0.00001;
lambda = 0.001;
lowestErrorRate = realmax;
lowestWij = Wij;
lowestWjk = Wjk;
currentErrorRate = realmax;
errControl = 0;
errThreshold = 5;
errPercent = 0.1;
trial = 0;
maxTrial = 50;

hiddenLayerInput=0;
hiddenLayerOutput=0;
outputLayerOutput=0;

%Learning loop
flag = 1;

while flag
    trial = trial+1;
    
    hiddenLayerInput = Wij' * X1;
    t_hli = Wij'*t_X1;
    %Apply sigmoid activation function to hidden layer
    hiddenLayerOutput = sigmoid(hiddenLayerInput);
    t_hlo = sigmoid(t_hli);

    %Add a bias row to hiddenLayerOutput
    biasrow = zeros (1, TrainingN);
    for i = 1:TrainingN
        biasrow(i)=1;
    end
    hiddenLayerOutput = cat(1,biasrow,hiddenLayerOutput(:,1:TrainingN));
    
    t_biasrow = zeros (1, TestingN);
    for i = 1:TestingN
        t_biasrow(i)=1;
    end
    t_hlo = cat(1,t_biasrow,t_hlo(:,1:TestingN));

    %Apply sigmoid activation function to output layer
    outputLayerOutput = softmax(Wjk, hiddenLayerOutput);
    t_olo = softmax(Wjk,t_hlo);
    
    %checking
    for i = 1:TrainingN
        [val,idx] = max(outputLayerOutput(:,i));
        predict(idx,i) = 1;
    end
    for i = 1:TestingN
        [val,idx] = max(t_olo(:,i));
        t_predict(idx,i) = 1;
    end
    %error rate
    
    err = 0;
    for i = 1:TrainingN
        if ~isequal(predict(:,i),t(:,i))
            err = err+1;
        end
    end
    errRate = err/TrainingN;
    
    %Plotting Training Data
    plot(trial,1-errRate,'o');
    hold on;
    
    t_err = 0;
    for i = 1:TestingN
        if ~isequal(t_predict(:,i),t_teach(:,i))
            t_err = t_err+1;
        end
    end
    t_errRate = t_err/TestingN;
    
    
    %Plotting Test Data
    %plot(trial,1-t_errRate,'o');
    %hold on;
    
    %Plotting Cross Entropy Loss Training Data
    %loss = crossEntropy(t, outputLayerOutput);
    %plot(trial,loss,'o');
    %hold on;
    
    %Plotting Cross Entropy Loss Test Data
    %loss = crossEntropy(t_teach, t_olo);
    %plot(trial,loss,'o');
    %hold on;
    
    
    
    %Terminating conditions
    if t_errRate > currentErrorRate
        errControl = errControl+1;
    else
        errControl = 0;
    end
    if errControl>errThreshold||t_err==0||trial>maxTrial||currentErrorRate<errPercent
        flag=0;
        break;
    end
    currentErrorRate=t_errRate;
    if (t_errRate<lowestErrorRate)
        lowestErrorRate = t_errRate;
        lowestWij = Wij;
        lowestWjk = Wjk;
    end
    %Calculating delta using backpropagation
    Dk_bp = t - outputLayerOutput;
    Wjk_Dk = Wjk*Dk_bp;
    Wjk_Dk = Wjk_Dk(2:end,:);
    Dj_bp = Wjk_Dk.*(sigmoidPrime(Wij'*X1));
    %Learn
    Wjk = Wjk + hiddenLayerOutput*Dk_bp'*learnRate;
    Wij = Wij + X1*Dj_bp'*learnRate;
    predict = zeros(C,TrainingN);
    t_predict = zeros(C,TestingN);
    
    
end

