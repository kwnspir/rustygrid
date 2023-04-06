use rand::Rng;

const GRID_SIZE: usize = 32;
const NUM_STATES: usize = GRID_SIZE * GRID_SIZE;
const NUM_ACTIONS: usize = 4; // Up, Down, Left, Right

fn state_to_coord(state: usize) -> (usize, usize) {
    (state / GRID_SIZE, state % GRID_SIZE)
}

fn coord_to_state(x: usize, y: usize) -> usize {
    x * GRID_SIZE + y
}

fn get_next_state_and_reward(
    state: usize,
    action: usize,
    gridworld: &Vec<Vec<char>>,
) -> (usize, isize) {
    let (mut x, mut y) = state_to_coord(state);

    match action {
        0 => x = x.saturating_sub(1), // Up
        1 => x = (x + 1).min(GRID_SIZE - 1), // Down
        2 => y = y.saturating_sub(1), // Left
        3 => y = (y + 1).min(GRID_SIZE - 1), // Right
        _ => (),
    }

    let next_state = coord_to_state(x, y);
    let reward = match gridworld[x][y] {
        'O' => -100,
        'G' => 100,
        _ => -1,
    };

    (next_state, reward)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Gridworld environment
    let mut gridworld = vec![vec!['-'; GRID_SIZE]; GRID_SIZE];

    for row in &mut gridworld {
        for cell in row.iter_mut() {
            if rng.gen_ratio(9, 10) {
                *cell = '-';
            } else {
                *cell = 'O';
            }
        }
    }

    gridworld[0][0] = 'S';
    gridworld[GRID_SIZE - 1][GRID_SIZE - 1] = 'G';

    // Initialize Q-table
    let mut q_table = vec![vec![0.0; NUM_ACTIONS]; NUM_STATES];

    // Hyperparameters
    let alpha = 0.1;
    let gamma = 0.99;
    let epsilon = 0.1;
    let num_episodes = 10000;

    // Q-learning algorithm
    for _ in 0..num_episodes {
        let mut state = 0;
        let mut done = false;

        while !done {
            let action = if rng.gen_bool(epsilon as f64) {
                rng.gen_range(0..NUM_ACTIONS)
            } else {
                q_table[state]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            };

            let (next_state, reward) = get_next_state_and_reward(state, action, &gridworld);
            q_table[state][action] += alpha * (reward as f64 + gamma * q_table[next_state].iter().cloned().fold(f64::MIN, f64::max) - q_table[state][action]);

            state = next_state;
            done = gridworld[state_to_coord(state)] == 'G';
        }
    }

    // Test the learned policy
    let mut state = 0;
    let mut path = vec![];

    while gridworld[state_to_coord(state)] != 'G' {
        path.push(state_to_coord(state));
        let action        = q_table[state]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let (next_state, _) = get_next_state_and_reward(state, action, &gridworld);
        state = next_state;
    }

    path.push(state_to_coord(state));
    println!("\nLearned path:");

    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            if path.contains(&(row, col)) {
                print!("x ");
            } else {
                print!("{} ", gridworld[row][col]);
            }
        }
        println!();
    }
}

