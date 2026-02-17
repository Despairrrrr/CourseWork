clc; clear; close all;

%% 1. Находим бифуркационное значение r (касание кривых) (рис)
syms u r;

% Уравнение неподвижной точки: f(u) = u
f = r * u^2 * (1 - log(1 + u));
eq1 = f == u;

% Условие бифуркации (касание): f'(u) = 1
df = diff(f, u);
eq2 = df == 1;

% Решаем систему
solution = solve([eq1, eq2], [u, r]);
r_bifurcation = double(solution.r);
u_bifurcation = double(solution.u);

disp(['Бифуркация происходит при r = ', num2str(r_bifurcation)]);
disp(['Соответствующее значение u = ', num2str(u_bifurcation)]);

%% 2. Строим график для трёх характерных значений r (рис1)
r_values = [2, r_bifurcation, 4]; % До, во время и после бифуркации
u = linspace(0, 2, 1000);         % Диапазон значений u

figure;
hold on;
grid on;

% Основная линия y = u
plot(u, u, 'k--', 'LineWidth', 1, 'DisplayName', 'y = u');

% Цвета для разных r
colors = ['b', 'r', 'g'];

% Рисуем кривые f(u) для каждого r
for i = 1:length(r_values)
    r_current = r_values(i);
    f_u = r_current * u.^2 .* (1 - log(1 + u));
    
    % Выбираем цвет и подпись
    if r_current < r_bifurcation
        legend_text = ['r = ', num2str(r_current), ' (1 точка)'];
    elseif r_current == r_bifurcation
        legend_text = ['r = ', num2str(r_current), ' (касание)'];
    else
        legend_text = ['r = ', num2str(r_current), ' (3 точки)'];
    end
    
    plot(u, f_u, colors(i), 'LineWidth', 1, 'DisplayName', legend_text);
    
    % Убраны все команды plot с маркерами точек пересечения
    % (это то, что отмечало точки пересечения)
end

%% 3. Настройка графика (рис1)
xlabel('u', 'FontSize', 12);
ylabel('f(u)', 'FontSize', 12);
title('Изменение количества неподвижных точек при разных r', 'FontSize', 14);
legend('Location', 'northwest');
% Выделяем область около бифуркации
xlim([0 1.5]);
ylim([0 1.5]);

hold off;
%% нахождение r* и построение графика зависимости производной
%в особых точках от значения r(Рис2)
clc;
clear;

% Значения параметра r
r_values = linspace(3.03, 5, 1000);

% Подготовка массивов
derivative_u2 = zeros(size(r_values));
derivative_u3 = zeros(size(r_values));

for i = 1:length(r_values)
    r = r_values(i);

    % Уравнение для поиска неподвижной точки
    fun = @(u) r*u.^2 .* (1 - log(1 + u)) - u;

    % Находим u_2* (примерно ближе к 0.4)
    u2 = fzero(fun, 0.4);
    derivative_u2(i) = r * u2 * ((u2 + 2)/(u2 + 1) - 2 * log(1 + u2));

    % Находим u_3* (примерно ближе к 1)
    u3 = fzero(fun, 1.0);
    derivative_u3(i) = r * u3 * ((u3 + 2)/(u3 + 1) - 2 * log(1 + u3));
end

% Построение графика
figure;
hold on;
plot(r_values, derivative_u2, 'b', 'DisplayName', 'u_2^*');
plot(r_values, derivative_u3, 'r', 'DisplayName', 'u_3^*');
plot(r_values, ones(size(r_values)), 'k--', 'DisplayName', '|f''(u)| = 1');
plot(r_values, -ones(size(r_values)), 'k--', 'HandleVisibility', 'off');
xlabel('r');
ylabel('f''(u^*)');
title('Значение производной в особой точке от параметра r');
legend('Location', 'best');
grid on;


%% (рис6) цикл длины 3
clc; clear;

% Устанавливаем значение параметра r (здесь подобрано так, чтобы цикл длины 3 существовал)
r = 5.05;

% Задаем правую часть рекурсивного уравнения f(u) = ru^2(1 - ln(1 + u))
f = @(u) r .* u.^2 .* (1 - log(1 + u));

% Тройное вложение функции f: f(f(f(u)))
% Необходимо для нахождения неподвижных точек третьего итератора — то есть точек, которые вернутся к себе через 3 итерации: fff(u) = u
fff = @(u) f(f(f(u)));

% Функция пересечения: ищем такие u, для которых f(f(f(u))) = u
% То есть: fff(u) - u = 0
intersection = @(u) fff(u) - u;

% Численное решение: ищем корни уравнения fff(u) = u
% В качестве начальных приближений берем несколько точек из интервала, где, согласно теории и рисункам, могут быть решения
u = fsolve(intersection, [0.5, 1.1, 1.5]);

% Выводим найденные неподвижные точки f(f(f(u))) = u
disp('Найденные точки цикла длины 3:');
disp(u);

% Строим график fff(u), y = u и отмечаем найденные точки
u_values = linspace(0, 1.8, 500);    % Значения u на отрезке [0, 1.8]
fff_values = fff(u_values);         % Значения fff(u)

plot(u_values, fff_values, 'b', ...  % График f(f(f(u)))
     u_values, u_values, 'r--', ...  % Прямая y = u
     u, fff(u), 'ro');               % Найденные точки цикла (пересечения)

xlabel('u');
ylabel('fff(u)');
%title('Поиск цикла длины 3: fff(u) = u');
legend('f(f(f(u))', 'y = u', 'Точки цикла длины 3', 'Location', 'best');
grid on;

%% (рис7)Поиск цикла длины 3
clc; clear;

r = 5.05;  % Значение параметра r

% Определяем функцию f(u)
f = @(u) r .* u.^2 .* (1 - log(1 + u));

% Определяем вложенную функцию третьего порядка: fff(u) = f(f(f(u)))
fff = @(u) f(f(f(u)));

% Ищем решения уравнения f(f(f(u))) = u
intersection = @(u) fff(u) - u;

% Начальные приближения для поиска корней
initial_guesses = [0.5, 1.1, 1.5];

% Массив для хранения найденных корней
u = zeros(size(initial_guesses));

% Настройки fsolve: не выводить сообщения
options = optimoptions('fsolve', 'Display', 'none');

% Вычисление неподвижных точек третьей итерации
for i = 1:length(initial_guesses)
    u(i) = fsolve(intersection, initial_guesses(i), options);
end

% Выводим найденные значения
disp('Найденные точки цикла длины 3:');
disp(u);

% Визуализация: f(u) и y = u
u_values = linspace(0, 1.8, 500);
f_values = f(u_values);

figure;
plot(u_values, f_values, 'b', 'LineWidth', 1);     % f(u)
hold on;
plot(u_values, u_values, 'r--', 'LineWidth', 1);   % y = u

% Ещё раз определяем fff и строим fff(u)
fff_values = fff(u_values);
plot(u_values, fff_values, 'b--', 'LineWidth', 1);             % f(f(f(u)))
plot(u, fff(u), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');% точки пересечения

% Рисуем лестничную диаграмму (cobweb) для каждой из найденных точек
for i = 1:length(u)
    u0 = u(i);
    for j = 1:10
        u1 = f(u0);
        plot([u0 u0], [u0 u1], 'k-', 'LineWidth', 1); % вертикальный сегмент
        plot([u0 u1], [u1 u1], 'k-', 'LineWidth', 1); % горизонтальный сегмент
        u0 = u1;
    end
end

xlabel('u');
ylabel('f(u)');
legend('f(u)', 'y = u', 'f(f(f(u)))', 'Циклические точки', 'Location', 'best');
grid on;
hold off;

%% (рис8)Поиск цикла длины 2
clc; clear;

r = 4.4;  % Значение параметра r

% Определяем f(u)
f = @(u) r .* u.^2 .* (1 - log(1 + u));

% Вложенная функция второго порядка: ff(u) = f(f(u))
ff = @(u) f(f(u));

% Ищем корни уравнения f(f(u)) = u
intersection = @(u) ff(u) - u;

% Начальные приближения
initial_guesses = [1, 1.5];

% Хранилище найденных корней
u = zeros(size(initial_guesses));
options = optimoptions('fsolve', 'Display', 'none');

% Вычисляем корни
for i = 1:length(initial_guesses)
    u(i) = fsolve(intersection, initial_guesses(i), options);
end

% Выводим найденные точки
disp('Найденные точки цикла длины 2:');
disp(u);

% Визуализация: f(u), y = u, ff(u)
u_values = linspace(0, 1.8, 500);
f_values = f(u_values);
ff_values = ff(u_values);

figure;
plot(u_values, f_values, 'b', 'LineWidth', 1);         % f(u)
hold on;
plot(u_values, u_values, 'r--', 'LineWidth', 1);       % y = u
plot(u_values, ff_values, 'g--', 'LineWidth', 1);      % f(f(u))
plot(u, ff(u), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r'); % Точки цикла

% Рисуем лестничную диаграмму (cobweb) для каждой найденной точки
for i = 1:length(u)
    u0 = u(i);
    for j = 1:10
        u1 = f(u0);
        u2 = f(u1);
        plot([u0 u0], [u0 u1], 'k-', 'LineWidth', 1);  % вверх
        plot([u0 u1], [u1 u1], 'k-', 'LineWidth', 1);  % вправо
        plot([u1 u1], [u1 u2], 'k-', 'LineWidth', 1);  % вверх
        plot([u1 u2], [u2 u2], 'k-', 'LineWidth', 1);  % вправо
        u0 = u2;
    end
end

xlabel('u');
ylabel('f(u)');
legend('f(u)', 'y = u', 'f(f(u))', 'Точки цикла', 'Location', 'best');
grid on;
hold off;
%% (рис9)
clc;
clear;

% Диапазон значений параметра r
r = linspace(0, 6, 1000);

% Количество итераций и шагов, которые мы визуализируем
num_iterations = 800;    % общее число итераций
num_display = 200;       % сколько последних точек отобразить

% Начальное значение u
u0 = 0.5;

% Подготовка графика
hold on;
for i = 1:length(r)
    % Инициализируем траекторию
    u = zeros(1, num_iterations);
    u(1) = u0;

    % Функция f(u) при данном r(i)
    f = @(u) r(i) * u^2 * (1 - log(1 + u));

    % Итерации отображения
    for n = 2:num_iterations
        u(n) = f(u(n - 1));
    end

    % Отображаем только последние значения после "разогрева"
    plot(r(i) * ones(1, num_display), u(end - num_display + 1:end), '.', 'MarkerSize', 1);
end

% Оформление графика
xlabel('r');
ylabel('u_{t+1}');
title('Бифуркационная диаграмма отображения');
hold off;
%% (рис 10)показатель ляпунова p(r)
% Параметры исследования
r_min = 0;
r_max = 6;
r_step = 0.01;
transient_steps = 1000; % Шаги для выхода на аттрактор
measure_steps = 1000;   % Шаги для вычисления показателя
u0 = 0.7;               % Начальное условие

% Инициализация
r_values = r_min:r_step:r_max;
lyapunov_exponents = zeros(size(r_values));

% Функция отображения
f = @(u, r) r * u^2 * (1 - log(1 + u));

% Производная отображения (аналитически вычисленная)
f_prime = @(u, r) r * u * ((u + 2)/(u + 1) - 2*log(1 + u));

% Основной цикл по значениям r
for k = 1:length(r_values)
    r = r_values(k);
    u = u0;
    sum_log_deriv = 0;
    valid_steps = 0;
    
    % Транзиентная фаза (выход на аттрактор)
    for i = 1:transient_steps
        u = f(u, r);
        % Проверка на расходимость
        if isnan(u) || isinf(u) || u < -0.999
            break;
        end
    end
    
    % Фаза измерения
    if ~(isnan(u) || isinf(u) || u < -0.999)
        for i = 1:measure_steps
            % Вычисление производной
            deriv = f_prime(u, r);
            
            % Обновление суммы логарифмов
            if abs(deriv) > 1e-10 % Исключаем сингулярные точки
                sum_log_deriv = sum_log_deriv + log(abs(deriv));
                valid_steps = valid_steps + 1;
            end
            
            % Переход к следующей точке
            u = f(u, r);
            
            % Проверка на расходимость
            if isnan(u) || isinf(u) || u < -0.999
                break;
            end
        end
    end
    
    % Вычисление показателя Ляпунова
    if valid_steps > 0
        lyapunov_exponents(k) = sum_log_deriv / valid_steps;
    else
        lyapunov_exponents(k) = NaN;
    end
end

% Построение графика
figure;
plot(r_values, lyapunov_exponents, 'b-', 'LineWidth', 1);
hold on;
plot([r_min r_max], [0 0], 'r--'); % Нулевая линия для сравнения
xlabel('Параметр r');
ylabel('Показатель Ляпунова p');
title('Зависимость показателя Ляпунова от параметра r');
grid on;
xlim([r_min r_max]);
legend('p(r)', 'Нулевой уровень');
%% (рис3) r менее порогового – две точки: одна устойчивая
r = 2;   u0 = 0.5;  X = 2;
plot_cobweb_my(r,u0,X,12,0,0);
%12-число итераций

%% (рис4) r между 3.0269 и r* – три точки, u2^* неуст., u3^* устойч.
r = 4;   u0 = 0.4; X = 1.4; u1 = 1.26; 
plot_cobweb_my(r,u0,X,4,u1,10);

%% (рис5) r > r* – обе нелинейные точки неустойчивы
r = 4.5; u0 = 0.35; X = 1.6; u1 = 1.32;
plot_cobweb_my(r,u0,X,4,u1,4);
%% Построение лесенки (для рисунков 4,5,6)

function y = f_my(r,u)
    y = r .* u.^2 .* (1 - log(1+u));
end

function y = fprime_my(r,u)
    y = r .* ( 2*u .* (1 - log(1+u)) - u.^2 ./ (1+u) );
end

%если передаваемых аргументов 4, то тогда это для u=0
%если больше, то второй и третий(7)
%с 5-6 это для нач приближения для u3
function plot_cobweb_my(r,u0,X,iters,u1,iters1)
%  r       – параметр
%  u0      – начальное значение
%  X       – правая граница по x (левую держим в  -0.1 для вида)
%  iters   – число итераций для лесенки
%
    if nargin<4, iters = 20; end
    figure; hold on; box on; grid on
    % --- оси
    u = linspace(-0.1,X,1200);
    plot(u,u,'k--','LineWidth',1.1)               % y = x
    plot(u,f_my(r,u),'Color',[0 0.45 0.8],'LineWidth',1.4) % y = f_r(u)

    % --- найти неподвижные точки (решаем f(u)-u=0)
    g = @(x) f_my(r,x)-x;
    guesses = linspace(0,X,6);
    fp = [];
    for s = guesses
        try
            root = fzero(g,s);
            if root>=0 && root<=X && all(abs(root-fp)>1e-4)
                fp(end+1)=root; 
            end
        catch
        end
    end

    % --- пометить неподвижные точки
    for xfp = fp
        stab = abs(fprime_my(r,xfp))<1;
        if stab
            plot(xfp,xfp,'ks','MarkerFaceColor','k','MarkerSize',7); % устойчивые – чёрные квадраты
        else
            plot(xfp,xfp,'ro','MarkerFaceColor','w','MarkerSize',7,'LineWidth',1.2); % неустойчивые – красные кружки
        end
    end

    % --- лесенка cobweb
    u_cur = u0;
    plot(u_cur,f_my(r, u_cur),'p','MarkerSize',10,'MarkerFaceColor','y') % начальная точка
    for k = 1:iters
        u_next = f_my(r,u_cur);
        % горизонталь к диагонали
        plot([u_cur u_next],[u_next u_next],'Color',[0 0.2 1]);
        % вертикаль к кривой
        plot([u_next u_next],[u_next f_my(r,u_next)],'Color',[0 0.2 1]);
        u_cur = u_next;
    end

    %если есть нач приближение для u3
    if u1 ~= 0
        u_cur = u1;
        plot(u_cur,f_my(r, u_cur),'p','MarkerSize',10,'MarkerFaceColor','y') % начальная точка
        for k = 1:iters1
            u_next = f_my(r,u_cur);
            % горизонталь к диагонали
            plot([u_cur u_next],[u_next u_next],'Color',[0.85 0 0]);
            % вертикаль к кривой
            plot([u_next u_next],[u_next f_my(r,u_next)],'Color',[0.85 0 0]);
            u_cur = u_next;
        end
    end
    %завершение
    xlabel('u'); ylabel('f(u)');
    %title(sprintf('Cobweb-plot: r = %.3g,  u^{0} = %.3g',r,u0));
    axis([-0.1 X -0.1 X]);
    legend({'y = x','y = f_r(u)'},'Location','SouthEast');
    hold off
end

%% Собственные числа в ненулевых фикс-точках для u1*(r), u2*(r)(11 рис)
clc; clear; close all;

% Модель
f_fix = @(r,u) r.*u.^2.*(1 - log(1+u)) - u;   % уравнение на фикс-точку u^*
phi   = @(u) u.*(1 - log(1+u));               % вспомогательная
uL = 1e-10; 
uR = exp(1) - 1 - 1e-10;                      % (0, e-1)

% Найдём r_min (точка касания)
[umax, negphi] = fminbnd(@(x)-phi(x), uL, uR);
r_min = 1/(-negphi);

% Диапазон r
rs = linspace(2.6, 10, 600);

% Ветви u^*(r)
U1 = nan(size(rs)); 
U2 = nan(size(rs));

for i = 1:numel(rs)
    r = rs(i);
    g = @(u) f_fix(r,u);

    % ищем все корни g(u)=0 на (0, e-1): сканирование + fzero
    uu = linspace(uL, uR, 3000);
    gu = g(uu);
    idx = find(gu(1:end-1).*gu(2:end) <= 0);   % смена знака

    roots_here = [];
    for k = idx
        a = uu(k); b = uu(k+1);
        if ~isfinite(g(a)) || ~isfinite(g(b)) || g(a)*g(b)>0, continue; end
        roots_here(end+1) = fzero(g, [a b]); %#ok<SAGROW>
    end
    roots_here = unique(round(roots_here,10));

    if numel(roots_here)==1
        U1(i) = roots_here;                    % кратный корень при r≈r_min
    elseif numel(roots_here)>=2
        U1(i) = roots_here(1); 
        U2(i) = roots_here(end);
    end
end

% Функция, возвращающая два собственных значения для данной (r,u*)
compute_lams = @(r,u) deal(NaN,NaN);  % заглушка для сигнатуры
% оформим как локальную вспомогательную функцию в конце файла
% см. функцию getLambdaPair(r,u) ниже

% Посчитаем λ для нижней ветви
lam1_u1 = nan(size(rs)); 
lam2_u1 = nan(size(rs));
for i = 1:numel(rs)
    if ~isnan(U1(i)) && U1(i)>0
        [lam1_u1(i), lam2_u1(i)] = getLambdaPair(rs(i), U1(i));
    end
end

% И для верхней ветви
lam1_u2 = nan(size(rs)); 
lam2_u2 = nan(size(rs));
for i = 1:numel(rs)
    if ~isnan(U2(i)) && U2(i)>0
        [lam1_u2(i), lam2_u2(i)] = getLambdaPair(rs(i), U2(i));
    end
end

% Рисуем (как у тебя: две панели). Можно рисовать и |λ|, если нужно.
figure('Color','w');

subplot(2,1,1); hold on; grid on; box on;
plot(rs, lam1_u1, 'b', 'LineWidth', 1.2);
plot(rs, lam2_u1, 'r', 'LineWidth', 1.2);
xline(r_min,'k--','r_{min}','LabelOrientation','horizontal');
xlabel('r'); ylabel('\lambda');
title('Собственные значения в ненулевой точке u_1^*(r)');
legend('\lambda_1','\lambda_2','Location','best');

subplot(2,1,2); hold on; grid on; box on;
plot(rs, lam1_u2, 'b', 'LineWidth', 1.2);
plot(rs, lam2_u2, 'r', 'LineWidth', 1.2);
xline(r_min,'k--','r_{min}','LabelOrientation','horizontal');
xlabel('r'); ylabel('\lambda');
title('Собственные значения в ненулевой точке u_2^*(r)');
legend('\lambda_1','\lambda_2','Location','best');

% ===== Локальная функция (внизу файла) =====
function [lam1,lam2] = getLambdaPair(r,u)
    if isnan(u) || u<=0
        lam1 = NaN; lam2 = NaN; return;
    end
    q = r*u^2/(1+u);
    if q <= 1
        s = sqrt(1 - q);
        lam1 = 1 + s;
        lam2 = 1 - s;
    else
        s = sqrt(q - 1);
        lam1 = 1 + 1i*s;
        lam2 = 1 - 1i*s;
        % если хочешь модуль, раскомментируй:
        % lam1 = abs(lam1); lam2 = abs(lam2);
    end
end


%% Система с запаздыванием — фазовый портрет (u,v)
clc, clear

r  = 1.1;   % параметр r
v0 = 0.9;   % u_0  (это v_t в модели с задержкой)
u0 = 0.7;   % u_1
N  = 15;    % число итераций

% Двумерное отображение
u = zeros(1, N);
v = zeros(1, N);
u(1) = u0;
v(1) = v0;

% Итерации траектории системы с запаздыванием:
% u_{t+1} = r * u_t^2 * (1 - ln(1 + v_t)),  v_{t+1} = u_t
for t = 1:N-1
    u(t+1) = r * u(t)^2 * (1 - log(1 + v(t)));   % u_{t+1}
    v(t+1) = u(t);                                % v_{t+1}
end

% Рисуем фазовый портрет (точками)
figure; hold on; grid on;
plot(u, v, 'r.', 'MarkerSize', 10);    % <-- именно это даёт такой график
xlabel('u'); ylabel('v');
title(['Фазовый портрет при r = ', num2str(r)]);

