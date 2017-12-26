function y = L1(w,lambda)
    [row,col] = size(w);
    ww=zeros(row,col);
    for i = 1:col
        sum=0;
        for j = 1:row
            sum = sum + w(j,i)^2;
        end
        for j = 1:row
            ww(j,i)=ww(j,i)/sqrt(sum);
        end
    end
    y = w - lambda*ww;
end