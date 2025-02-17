function plotNoiseReconstructionData( initialData, reconstructedData,...
    originalDenoisedData, showOriginalDenoisedData, filename )

x0 = 0;
y0 = 0;
width = 7;
height = 6;

fig1 = figure('Units','inches',...
        'Position',[x0 y0 width height],...
        'PaperPositionMode','auto');
set(gca,...
    'Units','normalized',...
    'Position',[.15 .2 .75 .7],...
    'FontUnits','points',...
    'FontWeight','bold',...
    'FontSize',9,...
    'FontName','Times');

plot( initialData(:, 1), initialData(:, 2), 'ro', 'DisplayName', 'Original Data' );
hold on;
plot( reconstructedData(:, 1), reconstructedData(:, 2), 'bo', "MarkerSize", 10,...\
    'MarkerFaceColor', 'b', 'DisplayName', 'Reconstructed Data' );
hold on;

if( showOriginalDenoisedData )
    plot( originalDenoisedData(:, 1), originalDenoisedData(:, 2), 'g', "LineWidth", 2,...\
        'DisplayName', 'Original Denoised Data' );
end

xlabel('{\boldmath$X$}',...
    'FontUnits','points',...
    'FontWeight','bold',...
    'FontSize',9,...
    'Interpreter', 'latex',...
    'FontName','Times');

ylabel('{\boldmath$Y$}',...
    'FontUnits','points',...
    'FontWeight','bold',...
    'FontSize',9,...
    'Interpreter', 'latex',...
    'FontName','Times');

legend();
print( fig1, filename, '-depsc2' );

end