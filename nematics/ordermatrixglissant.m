function Q=ordermatrixglissant(A,a)

%A=imread(d);
Q=[];

ii=1;
for i=1:floor(a/2):size(A,1)-a+1
    jj=1;
    for j=1:floor(a/2):size(A,2)-a+1
        
        B=A(i:i+a-1,j:j+a-1);
        
        
        q=sqrt((sum(sum(cos(2*B)))/(size(B,1)*size(B,2)))^2+(sum(sum(sin(2*B)))/(size(B,1)*size(B,2)))^2);
        %q=(sum(sum(cos(2*B)))/(size(B,1)*size(B,2)));
        
        
        % q=(sum(sum(cos(B)))/(size(B,1)*size(B,2)));
        
        %         Q(i:i+a-1,j:j+a-1)=q;
        
        Q(i:i+a-1,j:j+a-1)=q;
        jj=jj+1;
    end;
    ii=ii+1;
end;











% function Q=ordermatrixglissant(A,a)
% 
% %A=imread(d);
% Q=[];
% ii=1;
% for i=1:floor(a/4):size(A,1)-a+1
%     jj=1;
%     for j=1:floor(a/4):size(A,2)-a+1
%         
%         
% %         B=A(ii,jj);
%         B=A(i:i+a-1,j:j+a-1);
%         
%         q=sqrt((sum(sum(cos(2*B)))/(size(B,1)*size(B,2)))^2+(sum(sum(sin(2*B)))/(size(B,1)*size(B,2)))^2);
% %         q=sqrt((sum(sum(cos(B)))/(size(B,1)*size(B,2)))^2+(sum(sum(sin(B)))/(size(B,1)*size(B,2)))^2);
% %         q=(sum(sum(cos(2*B)))/(size(B,1)*size(B,2)));
%         
%         
%         % q=(sum(sum(cos(B)))/(size(B,1)*size(B,2)));
%         
%         Q(ii,jj)=q;
%         jj=jj+1;
%     end;
%     ii=ii+1;
% end;

