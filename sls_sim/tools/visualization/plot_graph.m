function plot_graph(adjMtx, nodeCoords, colour)
% Visualizes graph
%   adjMtx     : adjacency matrix
%   nodeCoords : x,y coordinates of each node (in order)
%   colour     : colour of nodes, either a letter or a RGB coordinate

gplot(adjMtx, nodeCoords, '-ok');

nodeDim = size(nodeCoords);
numNodes = nodeDim(1);

hold on
for node=1:numNodes;
    plot_vertex(node, nodeCoords, colour)
end
hold off

% dynamic axis limits
minCoords = min(nodeCoords);     maxCoords = max(nodeCoords); 
xlowerlim = minCoords(1) - 0.5;  xupperlim = maxCoords(1) + 0.5;
ylowerlim = minCoords(2) - 0.5;  yupperlim = maxCoords(2) + 0.5;

axis([xlowerlim xupperlim ylowerlim yupperlim])