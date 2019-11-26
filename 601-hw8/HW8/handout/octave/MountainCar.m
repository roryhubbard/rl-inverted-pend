classdef MountainCar < handle
    properties
        % Initial positions of box-car
        min_position = -1.2;
        max_position = 0.6;
        max_speed = 0.07;
        goal_position = 0.5;
        
        force = 0.001;
        gravity = 0.0025;
        

        % Actions = {0, 1, 2} for go left, do nothing, go right
        action_space = 3;
        state_space
            
        mode
        state
        random_gen
        
        % variables used conditionally on mode or render
        iht
        w
        viewer  % needed for render only
    end

    methods
        function obj = MountainCar(mode)
            if nargin == 0
                mode = [];
            end
            obj.mode = mode;
            if strcmp(mode,'tile') == 1
                obj.state_space = 2048;
            elseif strcmp(mode,'raw') == 1
                obj.state_space = 2;
            else
                error("Invalid environment mode. Must be tile or raw")
            end
            
            obj.seed();
            obj.reset();
        end
        
        function ret = transform(obj, state)
            % Normalize values to range from [0, 1] for use in transformations
            position = state(1);
            velocity = state(2);
            position = (position + 1.2) / 1.8;
            velocity = (velocity + 0.07) / 0.14;
            assert(0 <= position && position <= 1)
            assert(0 <= velocity && velocity <= 1)
            position = position * 2;
            velocity = velocity * 2;
            if strcmp(obj.mode, 'tile') == 1
                if isempty(obj.iht)
                    obj.iht = IHT(obj.state_space);
                end
                tiling = [tiles(obj.iht, 64, [position, velocity], 0),  ...
                        tiles(obj.iht, 64, position, 1),  ...
                        tiles(obj.iht, 64, velocity, 2)];
                ret = containers.Map(tiling, ones(1,length(tiling)));
                return;
            elseif strcmp(obj.mode,'raw') == 1
                ret = containers.Map(1:length(state), state);
                return;
            else
                error("Invalid environment mode. Must be tile or raw")
            end
        end
        
        function ret = seed(obj, seed)
            if nargin == 1
                seed = [];
            end
            obj.random_gen, ret = generate_random(seed);
            rand("state", ret);
        end
        
        function ret = reset(obj)
            obj.state = [rand*0.2 - 0.6, 0];
            ret = obj.transform(obj.state);
        end
        
        function ret = height(obj, xs)
            ret = sin(3 * xs) * 0.45 + 0.55;
        end
        
        function [ret, reward, done] = step(obj, action)
            assert(action == 1 || action == 2 || action == 3)
            position = obj.state(1);
            velocity = obj.state(2);
            velocity = velocity + (action-2)*obj.force + cos(3*position)*(-obj.gravity);
            velocity(velocity < -obj.max_speed) = -obj.max_speed;
            velocity(velocity > obj.max_speed) = obj.max_speed;
            position = position + velocity;
            position(position < obj.min_position) = -obj.min_position;
            position(position > obj.max_position) = obj.max_position;
            % Left of min_position is a wall
            if (position==obj.min_position && velocity<0)
                velocity = 0;
            end

            done = position >= obj.goal_position;
            reward = -1.0;

            obj.state = [position, velocity];
            ret = obj.transform(obj.state);
        end
    end
end

function [ret, seed] = generate_random(seed)
    if nargin == 0
        seed = [];
    end
    if ~isempty(seed) && not (isnumeric(seed) && 0 <= seed)
        error('Seed must be a non-negative integer or omitted.');
    end
    if isempty(seed)
        seed = randi(2^32 - 1);
    end
    ret = rand;
end