trajectories=deviationX_map~=0;

figure
subplot(1,2,1);
imagesc(trajectories);hold on;
subplot(1,2,2);
plot(ecartT,generalized_error,'b.');hold on;
center_of_mass=[sum(generalized_error)/size(generalized_error,2) sum(ecartT)/size(ecartT,2)];
hline(sum(generalized_error)/size(generalized_error,2),'--k');hold on;
vline(sum(ecartT)/size(ecartT,2),'--k');hold on;
xlabel('absolute standard deviation');ylabel('generalized error');
for i=2:size(position_proba,1)-1
    for j=2:size(position_proba,2)-1
        if trajectories(i,j)
            subplot(1,2,1);
            try
                delete(point);
                delete(square);
                delete(hcenter);
                delete(vcenter);
            end
            point=plot(j,i,'o','Color','g','markerfacecolor','g','MarkerSize',10);hold on;
            square=plot([j-1.5 j-1.5 j+1.5 j+1.5 j-1.5],[i-1.5 i+1.5 i+1.5 i-1.5 i-1.5],'-g');
            
            subplot(1,2,2);
            try 
                delete(deviation);
            end
            selection=[];
            for ii=1:size(position,2)
                if (position(1,ii)==i || position(1,ii)==i-1 || position(1,ii)==i+1) && (position(2,ii)==j || position(2,ii)==j+1 || position(2,ii)==j-1)
                    selection=[selection ii];
                end
            end
            title(['X = ',num2str(i),' | Y = ',num2str(j),' | N = ',num2str(size(selection,2)),' | Acc = ',num2str(100*sum([generalized_error(selection)<center_of_mass(1)].*[ecartT(selection)<center_of_mass(2)])/size(selection,2),3),' %']);
            deviation=plot(ecartT(selection),generalized_error(selection),'s','Color','r','markerfacecolor','g','MarkerSize',10);hold on;
            hcenter=hline(sum(generalized_error(selection))/size(selection,2),'k');
            vcenter=vline(sum(ecartT(selection))/size(selection,2),'k');
            pause;
        end
    end
end
