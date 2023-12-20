import numpy as np


def northwest_corner(supply, demand):
    """
    Função northwest_corner: Esta função aplica o método do canto noroeste para encontrar uma solução inicial viável para o 
    problema de transporte. Ela aloca as ofertas aos destinos, começando do canto superior esquerdo (noroeste) da matriz de 
    custos e avançando para baixo e para a direita.
    """
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    num_rows, num_cols = len(supply), len(demand)
    solution = np.zeros((num_rows, num_cols))

    i = j = 0
    while i < num_rows and j < num_cols:
        min_val = min(supply_copy[i], demand_copy[j])
        solution[i][j] = min_val
        supply_copy[i] -= min_val
        demand_copy[j] -= min_val

        if supply_copy[i] == 0 and i < num_rows - 1:
            i += 1
        elif demand_copy[j] == 0 and j < num_cols - 1:
            j += 1

        # Adiciona uma condição para sair do loop se ambas a oferta e a demanda na posição atual forem 0
        # e não houver mais linhas ou colunas para avançar
        if supply_copy[i] == 0 and demand_copy[j] == 0:
            if i == num_rows - 1 and j == num_cols - 1:
                break
            elif i < num_rows - 1:
                i += 1
            elif j < num_cols - 1:
                j += 1

    return solution


def verify_matrix_compatibility(cost_matrix, supply, demand):
    if cost_matrix.shape[0] != len(supply):
        raise ValueError("O número de origens na matriz de custos não corresponde ao tamanho da matriz de oferta.")
    if cost_matrix.shape[1] != len(demand):
        raise ValueError("O número de destinos na matriz de custos não corresponde ao tamanho da matriz de demanda.")


def calculate_weights(cost_matrix, solution):
    """
    Função calculate_weights: Esta função calcula os pesos das linhas e colunas com base na solução inicial e na matriz
    de custos. Os pesos são utilizados para calcular os custos reduzidos das células não alocadas.
    """
    num_rows, num_cols = cost_matrix.shape
    row_weights = np.zeros(num_rows)
    col_weights = np.zeros(num_cols)

    row_weights[0] = 0

    for i in range(num_rows):
        for j in range(num_cols):
            if solution[i][j] != 0:
                if row_weights[i] != np.inf and col_weights[j] == 0:
                    col_weights[j] = cost_matrix[i][j] - row_weights[i]
                elif col_weights[j] != np.inf and row_weights[i] == 0:
                    row_weights[i] = cost_matrix[i][j] - col_weights[j]

    for i in range(num_rows):
        for j in range(num_cols):
            if solution[i][j] != 0:
                if row_weights[i] == 0:
                    row_weights[i] = cost_matrix[i][j] - col_weights[j]
                if col_weights[j] == 0:
                    col_weights[j] = cost_matrix[i][j] - row_weights[i]

    return row_weights, col_weights


def check_for_improvement(cost_matrix, row_weights, col_weights):
    """
    Função check_for_improvement: Esta função verifica se existe uma oportunidade de melhoria na solução atual.
    Ela calcula o custo reduzido de cada célula não alocada e identifica a célula com o maior potencial de redução de custo.
    """
    num_rows, num_cols = cost_matrix.shape
    improvement = False
    min_value = 0
    min_cell = (0, 0)

    for i in range(num_rows):
        for j in range(num_cols):
            reduced_cost = cost_matrix[i][j] - row_weights[i] - col_weights[j]
            if reduced_cost < min_value:
                min_value = reduced_cost
                min_cell = (i, j)
                improvement = True

    return improvement, min_cell, min_value


def find_replacement_path(solution, start_cell):
    """
    Função find_replacement_path: Caso uma melhoria seja identificada, esta função encontra um caminho de substituição
    na matriz de alocação atual. O caminho de substituição é usado para ajustar a alocação atual de modo a reduzir o custo total.
    """
    num_rows, num_cols = solution.shape
    path = [start_cell]
    visited = np.zeros_like(solution, dtype=bool)

    def search(x, y, is_row):
        if visited[x, y]:
            return False
        visited[x, y] = True

        if (x, y) != start_cell and solution[x, y] > 0:
            path.append((x, y))
            return True

        if is_row:
            for j in range(num_cols):
                if j != y and search(x, j, False):
                    return True
        else:
            for i in range(num_rows):
                if i != x and search(i, y, True):
                    return True

        path.pop()
        return False

    for i in range(num_rows if start_cell[1] % 2 == 0 else num_cols):
        if (is_row := start_cell[1] % 2 == 0):
            if search(i, start_cell[1], not is_row):
                break
        else:
            if search(start_cell[0], i, is_row):
                break

    return path


def adjust_solution(solution, path):
    """
    Função adjust_solution: Esta função ajusta a solução atual com base no caminho de substituição encontrado,
    movendo as alocações para obter um custo total menor.
    """
    if not path:  # Check if the path is empty
        return solution  # Return the original solution if the path is empty

    min_val = min(solution[x, y] for x, y in path[1::2])

    for i, (x, y) in enumerate(path):
        if i % 2 == 0:
            solution[x, y] += min_val
        else:
            solution[x, y] -= min_val

    return solution


def balance_problem(cost_matrix, supply, demand):
    """
    Função balance_problem: Balanceia o problema de transporte adicionando uma origem ou destino fictício
    se a oferta total não for igual à demanda total.
    """
    total_supply = sum(supply)
    total_demand = sum(demand)

    if total_supply == total_demand:
        return cost_matrix, supply, demand

    if total_supply > total_demand:
        extra_demand = total_supply - total_demand
        demand = np.append(demand, extra_demand)
        cost_matrix = np.hstack((cost_matrix, np.zeros((cost_matrix.shape[0], 1))))
    elif total_demand > total_supply:
        extra_supply = total_demand - total_supply
        supply = np.append(supply, extra_supply)
        cost_matrix = np.vstack((cost_matrix, np.zeros((1, cost_matrix.shape[1]))))

    return cost_matrix, supply, demand


def read_transportation_problems(file):
    """
    Função read_transportation_problems: Esta função lê um arquivo de texto contendo vários problemas
    de transporte e os converte em matrizes de custo, oferta e demanda.
    """
    try:
        with open(file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Arquivo '{file}' não encontrado.")
        return []
    except IOError:
        print(f"Erro ao ler o arquivo '{file}'.")
        return []

    problems = content.strip().split('\n\n')
    results = []

    for problem in problems:
        lines = problem.split('\n')
        n_origins, n_destinations = map(int, lines[0].split())
        costs = [list(map(int, line.split())) for line in lines[1:n_origins+1]]
        offer = list(map(int, lines[n_origins+1].split()))
        demand = list(map(int, lines[n_origins+2].split()))
        results.append((np.array(costs), np.array(offer), np.array(demand)))
    
    for idx in range(len(results)):
        costs, offer, demand = results[idx]
        balanced_costs, balanced_offer, balanced_demand = balance_problem(costs, offer, demand)
        results[idx] = (balanced_costs, balanced_offer, balanced_demand)

    return results


def print_total_cost(cost_matrix, solution):
    """
    Função print_total_cost: Esta função é utilitária para imprimir o custo total.
    """
    total_cost = np.sum(cost_matrix * solution)
    print(f">> Custo total da solução: {total_cost:.1f}\n")


def print_custom_output(improvement, min_cell, min_value):
    """
    Função print_custom_output: Esta função é utilitária para imprimir as possíveis melhorias.
    """
    print("! Há uma possibilidade de melhoria: " + ("Sim" if improvement else "Não"))
    if improvement:
        print(f"- Local da melhoria: Célula (Origem {min_cell[0]+1}, Destino {min_cell[1]+1})")
        print(f"- Valor do custo reduzido mais negativo: {min_value:.1f}")
    print("\n")


def print_solution(solution):
    """
    Função print_solution: Esta função é utilitária para imprimir a solução atual.
    """
    num_origins, num_destinations = solution.shape
    for i in range(num_origins):
        for j in range(num_destinations):
            print(f"Origem {i+1} para Destino {j+1}: Transporta {solution[i][j]} unidades.")
    print("\n")


def solve_transportation_problems(file):
    """
    Função solve_transportation_problems: Esta função é o ponto de entrada principal do programa.
    Ela lê os problemas de transporte do arquivo, aplica as funções descritas acima para resolver
    cada problema e imprime os resultados.
    """
    problems = read_transportation_problems(file)

    if not problems:
        return []

    solutions = []

    for idx, (costs, offer, demand) in enumerate(problems, 1):
        try:
            verify_matrix_compatibility(costs, offer, demand)
        except ValueError as e:
            print(f"Erro no Problema {idx}: {e}")
            continue

        print(f"-> Resolvendo Problema {idx}...\n")
        initial_solution = northwest_corner(offer, demand)
        print("- Solução inicial com o método do Canto Noroeste:\n")
        print_solution(initial_solution)
        print_total_cost(costs, initial_solution)

        print(f"- Calculando solução ótima para o problema {idx}...\n")
        num_iterations = 0
        solution = initial_solution.copy()

        while True:
            num_iterations += 1
            print(f"> Iteração {num_iterations}...\n")

            row_weights, col_weights = calculate_weights(costs, solution)
            improvement, min_cell, min_value = check_for_improvement(costs, row_weights, col_weights)
            print_custom_output(improvement, min_cell, min_value)

            if not improvement:
                print("Solução ótima encontrada!\n")
                print_total_cost(costs, solution)
                break

            replacement_path = find_replacement_path(solution, min_cell)
            if not replacement_path:  # Verificar se o caminho está vazio
                print("# Nenhum caminho de substituição encontrado. Interrompendo a otimização. #\n")
                print_total_cost(costs, solution)  # Imprimir custo mesmo quando não há mais caminho
                break

            solution = adjust_solution(solution, replacement_path)
            print_total_cost(costs, solution)

        print("Solução Ótima:\n")
        print_solution(solution)
        solutions.append(solution)

    return solutions


if __name__ == "__main__":
    file_ = 'problemas.txt'
    solutions = solve_transportation_problems(file_)

