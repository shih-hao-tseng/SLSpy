function plot_time_traj(x, Bu, xDes)
% Plots time trajectories and errors (useful for trajectory tracking)
% Inputs
%    x, Bu : state and actuation values at nodes
%    xDes  : desired trajectory

figure;
Nx   = size(x, 1);
TMax = size(x, 2);

maxy = max([max(vec(x)) max(vec(Bu)) max(vec(xDes))]) + 2;
miny = min([min(vec(x)) min(vec(Bu)) max(vec(xDes))]) - 2;

err = abs(xDes - x);
maxe = max(vec(err)) * 1.1; 
mine = min(vec(err));
    
for node=1:Nx
    subplot(Nx, 2, node * 2 - 1);
    hold on
    stairs(1:TMax, x(node,:));
    stairs(1:TMax, xDes(node,:));
    stairs(1:TMax, Bu(node,:));
    set(gca,'XTickLabel',[]);
    ylabel(num2str(node));
    ylim([miny maxy]);
        
    subplot(Nx, 2, node * 2);
    stairs(1:TMax, err(node,:));
    set(gca,'XTickLabel',[]);
    ylabel(num2str(node));
    ylim([mine maxe]);
end

subplot(Nx, 2, Nx * 2 - 1)
legend('x', 'xDes', 'u');
xlabel('time step');
set(gca,'XTickLabelMode','auto');

subplot(Nx, 2, Nx * 2)
legend('error');
xlabel('time step');
set(gca,'XTickLabelMode','auto');
end

