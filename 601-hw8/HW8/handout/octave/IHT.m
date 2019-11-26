classdef IHT < handle
    properties
        overfullCount;
        dictionary;
        size;
    end

    methods
        function obj = IHT(sizeval)
            obj.size = sizeval;
            obj.overfullCount = 0;
            obj.dictionary = containers.Map;
        end
        
        function dict_len = count(obj)
            dict_len = obj.dictionary.Count;
        end
        
        function fullp_check = fullp(obj)
            fullp_check = obj.dictionary.Count >= obj.size;
        end
        
        function ret = getindex(obj, obj2, readonly)
            if nargin == 2
                readonly = false;
            end
            d = obj.dictionary;
            if isKey(d, sprintf('%d.',obj2))
                ret = d(sprintf('%d.',obj2));
                return;
            elseif readonly
                ret = [];
                return;
            end

            count = obj.count();

            if count >= obj.size
                if obj.overfullCount == 0
                    disp('IHT full, starting to allow collisions');
                end
                obj.overfullCount = obj.overfullCount + 1;
                ret = mod(basehash(obj),obj.size);
            else
                d(sprintf('%d.',obj2)) = count;
                ret = count;
            end
        end
            
        
        
    end
end
