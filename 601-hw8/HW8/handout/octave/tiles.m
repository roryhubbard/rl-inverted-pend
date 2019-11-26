function ret = tiles(iht, numtilings, floats, ints, readonly)

    if nargin < 5
        readonly = false;
    end
    if nargin < 4
        ints = [];
    end
    % returns num-tilings tile indices corresponding to the floats and ints
    
    qfloats = zeros(1,length(floats));
    for f = 1:length(floats)
        qfloats(f) = floor(floats(f) * numtilings);
    end

    Tiles = [];
    
    for tiling = 1:numtilings
        tilingX2 = tiling * 2;
        coords = tiling;
        b = tiling;
        for q = qfloats
            coords = [coords, floor((q + b) / numtilings)];
            b = b + tilingX2;
        end
        coords = [coords, ints];
        Tiles = [Tiles, hashcoords(coords, iht, readonly)];
    ret = Tiles;
    end
end