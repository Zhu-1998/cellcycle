%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Code for Landscape Visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
close all;


dir = 'G:\cell_cycle\landscape\WT\';
dor = 'grid=100, D=0.1';

grid=100;
D = 0.1;


%Enter input file name below
filename1= [dir, 'Xgrid.csv'];
Xgrid = readmatrix(filename1);

filename2= [dir, 'Ygrid.csv'];
Ygrid = readmatrix(filename2);

filename3= [dir, 'pot_U.csv'];
pot_U = readmatrix(filename3);

filename4= [dir, 'mean_Fx.csv'];
mean_Fx = readmatrix(filename4);

filename5= [dir, 'mean_Fy.csv'];
mean_Fy = readmatrix(filename5);

filename6= [dir, 'p_tra.csv'];
p_tra = readmatrix(filename6);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot landscape %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U = pot_U';
P = p_tra';
mean_Fx = mean_Fx';
mean_Fy = mean_Fy';

l = figure(1);
U(~isfinite(U))=21.2;
U(U>20)=20;
surf(Xgrid, Ygrid, U, 'LineStyle', '-', 'FaceColor', 'interp', 'FaceAlpha', 0.9);

xlim([-3.950297451019287, 11.845176315307617]) %cellcycle
ylim([-0.0980034828186036, 11.81999454498291]) %cellcycle
% zlim([3, 20]) %cellcycle
xticks([-2, 2, 6, 10])
yticks([1, 4, 7, 10])
zticks([5, 20])


xlabel('UMAP1');
ylabel('UMAP2');
zlabel('Potential');
colormap([jet(256)]);
set(gca, 'FontName', 'Arial')
set(gca,'FontSize',20, 'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1)
pbaspect([1 1 1])
% title(dor);
xlabel('UMAP1', 'Rotation', -22, 'Position',[7.271679428895709,-1.940801224249185,-3.557141103914375]);  %noise
ylabel('UMAP2', 'Rotation', 55, 'Position', [14.826168082398397,3.413625726717726,-4.139968097134329]);
set(gca, 'OuterPosition', [0.85697619047619,0.098095238095238,0.038095238095238,0.815000000000001]) %perturbation
set(gca, 'InnerPosition', [0.110357142857143,0.150476190476191,0.775,0.815])
set(gca, 'Position', [0.110357142857143,0.150476190476191,0.775,0.815])

set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02])
set(gca, 'LineWidth', 2, 'Color', [0 0 0])
set(gca, 'XColor', [0.00 0.00 0.00])
set(gca, 'YColor', [0.00 0.00 0.00])
set(gca, 'ZColor', [0.00 0.00 0.00])
box on 

% colorbar
colorbar('Ticks', [6, 10, 14, 18], 'TickLength', 0.02, 'TickDirection', 'out', 'FontSize', 20, 'Color', [0 0 0], 'LineWidth', 2, 'Position', [0.85697619047619,0.098095238095238,0.038095238095238,0.815000000000001])

alpha(1)
shading interp
lighting gouraud;
set(gca, 'color', 'white')

view([30, 45])
set(gca, 'color', 'white')
hold on
% contour(Xgrid, Ygrid, U, 20)
contour(Xgrid, Ygrid, U)

% [X,Y] = meshgrid(-4:16/99:12,-1:13/99:12); %noise
[X,Y] = meshgrid(-4:19/99:15,-1:13/99:12);
surfc(Xgrid, Ygrid, U)
surfc(X, Y, U, 20)
surf(Xgrid, Ygrid, 0+0*U, U)
shading interp
colormap([jet(256)]);

hold on
mesh(Xgrid(1:2:end, 1:2:end), Ygrid(1:2:end, 1:2:end), U(1:2:end, 1:2:end)+0.2, 'LineStyle', '-','LineWidth', 0.6, 'EdgeColor', 'k', 'FaceColor', 'none')

hold on
saveas( l, [dir, 'landscape.fig']); 
print(l, [dir, 'landscape.tif'],'-r600','-dtiff');
print(l, '-r600', '-dpdf', [dir, 'landscape.pdf']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot flux %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = figure(2);
pcolor(Xgrid,Ygrid,U);
shading interp;
colormap([jet(256)]);
xlabel('UMAP1');
ylabel('UMAP2');
set(gca, 'FontName', 'Arial')
set(gca,'FontSize',20, 'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1)
pbaspect([1 1 1])
set(gca, 'color', 'white');
set(gca,'TickDir', 'out', 'TickLength', [0.02 0])
% title([dor,', flux'])
%title(dor)
set(gca, 'LineWidth', 2, 'Color', [0 0 0])
set(gca, 'XColor', [0.00 0.00 0.00])
set(gca, 'YColor', [0.00 0.00 0.00])
set(gca, 'ZColor', [0.00 0.00 0.00])
xlim([-3.950297451019287, 11.845176315307617]) %cellcycle
ylim([-0.0980034828186036, 11.81999454498291]) %cellcycle
xticks([-2, 2, 6, 10])
yticks([1, 4, 7, 10])
colorbar
box off
colorbar('Ticks', [5, 10, 15, 20], 'TickLength', 0.02, 'TickDirection', 'out', 'FontSize', 20, 'Color', [0 0 0], 'LineWidth', 2)
hold on;

% P(P<=exp(-14))=0;
dx = (max(max(Xgrid))-min(min(Xgrid)))/grid;
dy = (max(max(Ygrid))-min(min(Ygrid)))/grid;
[GUx,GUy] = gradient(U,dx,dy);
[GPx,GPy] = gradient(P,dx,dy);
Jx = mean_Fx.*P - D*GPx ;
Jy = mean_Fy.*P - D*GPy ;

mg = 1:5:grid;
ng = mg;
E=Jy.^2+Jx.^2;
JJx=Jx./(sqrt(E)+eps);
JJy=Jy./(sqrt(E)+eps);
quiver(Xgrid(mg,ng),Ygrid(mg,ng),JJx(mg,ng),JJy(mg,ng),0.5,'color','k', 'LineWidth',1);
hold on;

saveas( f, [dir, 'flux_18.fig']); 
saveas( f, [dir, 'flux_v.fig']); 
print(f, [dir, 'flux.tif'],'-r600','-dtiff');
print(f, '-r600', '-dpdf', [dir, 'flux_15.pdf']);
print(f, '-r600', '-dpdf', [dir, 'flux.pdf']);

hold on;
% quiver(Xgrid(mg,ng),Ygrid(mg,ng),FFgradx(mg,ng),FFgrady(mg,ng),0.5,'color','w', 'LineWidth',1);
% saveas(f, [dir, 'flux_gradient.fig']); 
% print(f, '-r600', '-dpdf', [dir, 'flux_gradient.pdf']);
% print(f, [dir, 'flux_gradient.tif'],'-r600','-dtiff');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot F %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = figure(3);
pcolor(Xgrid,Ygrid,U);
shading interp;
colormap([jet(256)]);
xlabel('UMAP1');
ylabel('UMAP2');
set(gca, 'FontName', 'Arial')
set(gca,'FontSize',20, 'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1)
pbaspect([1 1 1])
set(gca, 'color', 'white');
set(gca,'TickDir','out')
%title([dor, ', <F>'])
set(gca, 'LineWidth', 2)
box off
xlim([-3.950297451019287, 11.845176315307617]) %cellcycle
ylim([-0.0980034828186036, 11.81999454498291]) %cellcycle
xticks([-2, 2, 6, 10])
yticks([1, 4, 7, 10])
colorbar
box off
colorbar('Ticks', [8, 10, 12, 14, 16], 'TickLength', 0.02, 'TickDirection', 'out', 'FontSize', 20, 'Color', [0 0 0], 'LineWidth', 2)
hold on;

EE=mean_Fx.^2+mean_Fy.^2;
FFx=mean_Fx./(sqrt(EE)+eps);
FFy=mean_Fy./(sqrt(EE)+eps);
quiver(Xgrid(mg,ng),Ygrid(mg,ng),FFx(mg,ng),FFy(mg,ng),0.5,'color','k' ,'LineWidth',1);

saveas( m, [dir, 'F.fig']);   
print(m, [dir, 'mean_F.tif'],'-r600','-dtiff');
print(m, '-r600', '-dpdf', [dir, 'F.pdf']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot F-grad %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = figure(4);
pcolor(Xgrid,Ygrid,U);
shading interp;
colormap([jet(256)]);
xlabel('UMAP1');
ylabel('UMAP2');
set(gca, 'FontName', 'Arial')
set(gca,'FontSize',22, 'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1)
pbaspect([1 1 1])
set(gca, 'color', 'white');
set(gca,'TickDir', 'out', 'TickLength', [0.02 0])
%title([dor, ', F-gradient'])
set(gca, 'LineWidth', 2, 'Color', [0 0 0])
set(gca, 'XColor', [0.00 0.00 0.00])
set(gca, 'YColor', [0.00 0.00 0.00])
set(gca, 'ZColor', [0.00 0.00 0.00])
xlim([-3.950297451019287, 11.845176315307617]) %cellcycle
ylim([-0.0980034828186036, 11.81999454498291]) %cellcycle
xticks([-2, 2, 6, 10])
yticks([1, 4, 7, 10])
box off

colorbar('Ticks', [8, 10, 12, 14], 'TickLength', 0.02, 'TickDirection', 'out', 'FontSize', 24, 'Color', [0 0 0], 'LineWidth', 2)
hold on;

Fgradx = -D*GUx;
Fgrady = -D*GUy;
EEE=Fgradx.^2+Fgrady.^2;
FFgradx=Fgradx./(sqrt(EEE)+eps);
FFgrady=Fgrady./(sqrt(EEE)+eps);
quiver(Xgrid(mg,ng),Ygrid(mg,ng),FFgradx(mg,ng),FFgrady(mg,ng),0.5,'color','k', 'LineWidth',1);

hold on;
saveas( g, [dir, 'grad_F.fig']);  
print(g, [dir, 'grad_F.tif'],'-r600','-dtiff');
print(g, '-r600', '-dpdf', [dir, 'grad_F.pdf']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





