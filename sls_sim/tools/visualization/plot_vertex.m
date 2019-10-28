function plot_vertex(node, nodeCoords, colour)
% Adds colour & label to specified vertex / node
%   node       : the node we're plotting
%   nodeCoords : x,y coordinates of each node (in order)
%   colour     : colour of node, either a letter or a RGB coordinate

txtOffset = 0.08; % how much to offset text labels

hold on
% make node a different colour
plot(nodeCoords(node,1), nodeCoords(node,2), ...
     'o','MarkerSize', 10, 'MarkerFaceColor',colour,'MarkerEdgeColor','k');
% label node    
text(nodeCoords(node,1)+txtOffset, nodeCoords(node,2)+txtOffset, num2str(node));
hold off