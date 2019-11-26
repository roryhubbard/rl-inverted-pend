function ret = hashcoords(coordinates, m, readonly)
    if nargin == 2
        readonly = false;
    end
    if isa(m,'IHT') == 1
        ret = m.getindex(coordinates, readonly);
    end
    if isempty(m)
        ret = coordinates; % for debugging purposes
    end
    ret = ret + 1;
end