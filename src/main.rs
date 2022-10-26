// GENTLE INTRODUCTION TO RUST: https://stevedonovan.github.io/rust-gentle-intro/
// MAXIMIZE PERFORMANCE WHEN COMPILING: https://nnethercote.github.io/perf-book/build-configuration.html
// RAYON FOR PARALLELIZABLE PROBLEMS:  https://docs.rs/rayon/latest/rayon/

// PROFILING: cargo flamegraph --dev
// RUN OPTIMIZED: cargo run --release
// TIME: time cargo run --release 205 12 0.75 100_000
// FIND COMMON ERRORS: cargo clippy

// TODO: Test limiting the number of multi-vertex cliques to one fewer than the best we've found
//       and then forcing remaining vertices into existing multi-vertex cliques,
//       removing conflicting vertices.

// TODO: Explore combining iterated greedy with Tabu.

// Both of the above to-do's are discussed in:
// A survey of local search methods for graph coloring, by Galinier & Hertz

// vcc = vertex clique cover
// mis = maximum independent set
// ..._bv = bitvector (BitVec)
// ..._ct = count (usize)
// force compile

use bitvec_simd::BitVec; // https://docs.rs/bitvec_simd/0.20.5/bitvec_simd/struct.BitVecSimd.html
use smallvec::{smallvec, SmallVec}; // https://docs.rs/smallvec/1.10.0/smallvec/struct.SmallVec.html
use std::env;
use std::time::Instant;

// The neighbors of a clique are those vertices that are not in the clique,
// and are adjacent to every vertex in the clique.
struct Clique {
  members_bv: BitVec,
  members_ct: usize,
  members: SmallVec<[usize; 256]>,
  neighbors_bv: BitVec,
  length: usize,
  id: usize,
  is_active: bool,
  has_neighbors: bool,
}

// A clique has at least one member, and at least zero neighbors.
// A clique with exactly one member is also referred to as a vertex or node.
impl Clique {
  fn new(num_vertices: usize, id: usize) -> Clique {
    Clique {
      members_bv: BitVec::zeros(num_vertices),
      members_ct: 1,
      members: smallvec![id],
      neighbors_bv: BitVec::zeros(num_vertices),
      length: num_vertices,
      id,
      is_active: true,
      has_neighbors: false,
    }
  }

  fn to_string(&self) -> String {
    let mut ret_str = String::new();
    for i in 0..self.length {
      if self.members_bv.get(i) == Some(true) {
        ret_str += "\u{25AA}";
      } else if self.neighbors_bv.get(i) == Some(true) {
        ret_str += "\u{25AB}";
      } else {
        ret_str += "\u{2B1D}";
      }
    }
    if !self.is_active {
      ret_str += " I";
    } else {
      ret_str += &(" ".to_owned() + &self.members_ct.to_string());
    }
    ret_str
  }
}

struct CliqueMaker {
  id: usize,
  length: usize,
}

impl CliqueMaker {
  fn new(num_vertices: usize) -> CliqueMaker {
    CliqueMaker {
      id: 0,
      length: num_vertices,
    }
  }

  // Returns a new clique with one member (incrementing which node is
  // in the clique), and no neighbors
  fn make_clique(&mut self) -> Clique {
    let mut ret_clique: Clique = Clique::new(self.length, self.id);
    ret_clique.members_bv.set(self.id, true);
    self.id += 1;
    ret_clique
  }

  fn get_copy_of_clique(&self, clique_to_copy: &Clique) -> Clique {
    let mut ret_clique: Clique = Clique::new(clique_to_copy.length, clique_to_copy.id);
    transcribe_clique_onto_clique(clique_to_copy, &mut ret_clique);
    ret_clique
  }
}

fn transcribe_clique_onto_clique(source_clique: &Clique, target_clique: &mut Clique) {
  target_clique.members_bv.set_all_false();
  target_clique.members.clear();
  if source_clique.members_ct == 1 {
    target_clique.members_bv.set(source_clique.members[0], true);
    target_clique.members.push(source_clique.members[0]);
  } else {
    target_clique
      .members_bv
      .or_inplace(&source_clique.members_bv);
    target_clique
      .members
      .extend_from_slice(&source_clique.members);
  }
  target_clique.members_ct = source_clique.members_ct;
  target_clique.neighbors_bv.set_all_false();
  target_clique
    .neighbors_bv
    .or_inplace(&source_clique.neighbors_bv);
  target_clique.length = source_clique.length;
  target_clique.id = source_clique.id;
  target_clique.is_active = source_clique.is_active;
  target_clique.has_neighbors = source_clique.has_neighbors;
}

struct Graph {
  size: usize,
  vertices: SmallVec<[Clique; 256]>,
  cliques: SmallVec<[Clique; 256]>,
  max_active_clique_index: usize,
  utility_bv: BitVec,
}

impl Graph {
  fn new(num_vertices: usize) -> Graph {
    let mut clique_maker = CliqueMaker::new(num_vertices);
    let mut vertices_vec: SmallVec<[Clique; 256]> = smallvec![];
    let mut cliques_vec: SmallVec<[Clique; 256]> = smallvec![];

    for _i in 0..num_vertices {
      let vertex = clique_maker.make_clique();
      let clique = clique_maker.get_copy_of_clique(&vertex);
      vertices_vec.push(vertex);
      cliques_vec.push(clique);
    }

    Graph {
      size: num_vertices,
      vertices: vertices_vec,
      cliques: cliques_vec,
      max_active_clique_index: num_vertices - 1,
      utility_bv: BitVec::zeros(num_vertices),
    }
  }

  fn transfer_compatible_vertices(
    clique_into: &mut Clique,
    clique_from: &mut Clique,
    utility_bv: &mut BitVec,
    vertices_vec: &SmallVec<[Clique; 256]>,
  ) {
    if !clique_into.has_neighbors {
      return;
    }

    // clear utility_bv
    utility_bv.set_all_false();

    // set utility_bv to be true for all transferrable vertices
    utility_bv.or_inplace(&clique_into.neighbors_bv);
    utility_bv.and_inplace(&clique_from.members_bv);
    if utility_bv.none() {
      return;
    }

    // update members_bv for both cliques
    clique_into.members_bv.or_inplace(utility_bv);
    clique_from.members_bv.xor_inplace(utility_bv);

    // update members & neighbors_bv for both cliques
    clique_from.neighbors_bv.set_all_true();
    for i in (0..clique_from.members_ct).rev() {
      if utility_bv.get_unchecked(clique_from.members[i]) {
        clique_into
          .neighbors_bv
          .and_inplace(&vertices_vec[clique_from.members[i]].neighbors_bv);
        clique_into.members.push(clique_from.members.swap_remove(i));
        clique_from.members_ct -= 1;
        clique_into.members_ct += 1;
      } else {
        clique_from
          .neighbors_bv
          .and_inplace(&vertices_vec[clique_from.members[i]].neighbors_bv);
      }
    }

    if clique_from.members_ct == 0 {
      clique_from.neighbors_bv.set_all_false();
      clique_from.is_active = false;
    } else {
      // If nothing else, it has some neighbors in clique_into
      clique_from.has_neighbors = true;
    }

    if clique_into.neighbors_bv.none() {
      clique_into.has_neighbors = false;
    }
  }

  fn shuffle_active_cliques(&mut self) {
    fastrand::shuffle(&mut self.cliques[0..(self.max_active_clique_index + 1)]);
  }

  fn reverse_active_cliques(&mut self) {
    self.cliques[0..(self.max_active_clique_index + 1)].reverse();
  }

  fn vcc_greedy(&mut self) {
    // Try to merge every active pair of cliques
    for i in 0..self.max_active_clique_index {
      if !self.cliques[i].is_active {
        continue;
      }
      for j in (i + 1)..(self.max_active_clique_index + 1) {
        if !self.cliques[j].is_active {
          continue;
        }
        let (cliques_before_j, cliques_from_j) = self.cliques.split_at_mut(j);
        let cliques_i: &mut Clique = &mut cliques_before_j[i];
        let cliques_j: &mut Clique = &mut cliques_from_j[0];
        Self::transfer_compatible_vertices(
          cliques_i,
          cliques_j,
          &mut self.utility_bv,
          &self.vertices,
        );
      }
    }
    // update cliques so active cliques precede inactive cliques
    let mut i = 0;
    let mut j = self.max_active_clique_index;
    let mut new_max_active_clique_index = 0;
    loop {
      if i > j {
        break;
      }
      if self.cliques[i].is_active {
        new_max_active_clique_index = i;
        i += 1;
      } else if self.cliques[j].is_active {
        self.cliques.swap(i, j);
        new_max_active_clique_index = i;
        i += 1;
        j -= 1;
      } else {
        j -= 1;
      }
    }
    self.max_active_clique_index = new_max_active_clique_index;
  }

  fn vcc_iterated_greedy(&mut self, reverse_fraction: f64) {
    if fastrand::f64() < reverse_fraction {
      self.reverse_active_cliques();
    } else {
      self.shuffle_active_cliques();
    }
    self.vcc_greedy();
  }

  fn vcc_run_iterations_to_target(
    &mut self,
    num_iterations: usize,
    target: usize,
    reverse_fraction: f64,
  ) -> bool {
    let mut pri_cliques = self.max_active_clique_index + 1;
    let mut current = Instant::now();
    for i in 1..(num_iterations + 1) {
      self.vcc_iterated_greedy(reverse_fraction);
      if i % 1_000_000 == 0 || self.max_active_clique_index + 1 < pri_cliques {
        println!(
          "Iteration {:0>3}_{:0>3}_{:0>3}: {} -> {} ({:?})",
          (i % 1_000_000_000) / 1_000_000,
          (i % 1_000_000) / 1_000,
          i % 1000,
          pri_cliques,
          self.max_active_clique_index + 1,
          current.elapsed()
        );
        current = Instant::now();
        pri_cliques = self.max_active_clique_index + 1;
        if self.max_active_clique_index + 1 <= target {
          return true;
        }
      }
    }
    false
  }

  fn conform_cliques_to_vertices(&mut self) {
    for i in 0..self.size {
      transcribe_clique_onto_clique(&self.vertices[i], &mut self.cliques[i]);
    }
    self.max_active_clique_index = self.size - 1;
  }

  fn to_vertex_string(&self) -> String {
    let mut ret_str = String::new();
    for i in 0..(self.size) {
      ret_str += &self.vertices[i].to_string();
      ret_str += "\n";
    }
    ret_str
  }

  fn to_string(&self) -> String {
    let mut ret_str = String::new();
    for i in 0..(self.max_active_clique_index + 1) {
      ret_str += &self.cliques[i].to_string();
      ret_str += "\n";
    }
    ret_str
  }
}

fn get_random_graph(num_vertices: usize, edge_probability: f64) -> Graph {
  let mut ret_graph = Graph::new(num_vertices);
  let mut edge_candidates_remaining = num_vertices * (num_vertices - 1) / 2;
  let mut edges_remaining = (edge_candidates_remaining as f64 * edge_probability) as usize;
  for i in 0..(ret_graph.size - 1) {
    for j in (i + 1)..(ret_graph.size) {
      if fastrand::f64() < (edges_remaining as f64) / (edge_candidates_remaining as f64) {
        edges_remaining -= 1;
        ret_graph.vertices[i].neighbors_bv.set(j, true);
        ret_graph.vertices[j].neighbors_bv.set(i, true);
      }
      edge_candidates_remaining -= 1;
    }
  }
  for i in 0..(ret_graph.size) {
    if ret_graph.vertices[i].neighbors_bv.any() {
      ret_graph.vertices[i].has_neighbors = true;
    }
  }
  ret_graph.conform_cliques_to_vertices();
  ret_graph.shuffle_active_cliques();
  ret_graph
}

fn get_random_graph_with_k_cliques(
  num_vertices: usize,
  num_cliques: usize,
  edge_probability: f64,
) -> Graph {
  if num_cliques == 0 {
    return get_random_graph(num_vertices, edge_probability);
  }

  let mut ret_graph = Graph::new(num_vertices);
  let mut edge_candidates_remaining = num_vertices * (num_vertices - 1) / 2;
  let mut edges_remaining = (edge_candidates_remaining as f64 * edge_probability) as usize;

  let reserved_edges =
    num_cliques * (num_vertices / num_cliques) * (num_vertices / num_cliques - 1) / 2
      + (num_vertices % num_cliques) * (num_vertices / num_cliques);
  edge_candidates_remaining -= reserved_edges;
  if reserved_edges > edges_remaining {
    edges_remaining = 0;
  } else {
    edges_remaining -= reserved_edges;
  }

  for i in 0..(ret_graph.size - 1) {
    for j in (i + 1)..(ret_graph.size) {
      if i % num_cliques == j % num_cliques {
        ret_graph.vertices[i].neighbors_bv.set(j, true);
        ret_graph.vertices[j].neighbors_bv.set(i, true);
      } else if fastrand::f64() < (edges_remaining as f64) / (edge_candidates_remaining as f64) {
        edges_remaining -= 1;
        ret_graph.vertices[i].neighbors_bv.set(j, true);
        ret_graph.vertices[j].neighbors_bv.set(i, true);
      }

      if i % num_cliques != j % num_cliques {
        edge_candidates_remaining -= 1;
      }
    }
  }
  for i in 0..(ret_graph.size) {
    if ret_graph.vertices[i].neighbors_bv.any() {
      ret_graph.vertices[i].has_neighbors = true;
    }
  }
  ret_graph.conform_cliques_to_vertices();
  ret_graph
}

fn clear_screen() {
  print!("\x1B[2J\x1B[1;1H");
}

fn main() {
  let args: Vec<String> = env::args().collect();
  let num_vertices: usize = args[1].parse().unwrap();
  let num_cliques: usize = args[2].parse().unwrap();
  let edge_fraction: f64 = args[3].parse().unwrap();
  let max_iterations_str: String = args[4].parse().unwrap();
  let max_iterations: usize = max_iterations_str.replace('_', "").parse().unwrap();
  let reverse_fraction: f64 = args[5].parse().unwrap();
  clear_screen();
  println!(
    "cargo run --release {} {} {} {} {}",
    num_vertices, num_cliques, edge_fraction, max_iterations_str, reverse_fraction
  );
  let mut g = get_random_graph_with_k_cliques(num_vertices, num_cliques, edge_fraction);
  let mut best_result: usize = num_vertices;
  loop {
    if g.vcc_run_iterations_to_target(max_iterations, num_cliques, reverse_fraction) {
      println!("\n{}", g.to_string());
      g = get_random_graph_with_k_cliques(num_vertices, num_cliques, edge_fraction);
    } else {
      if g.max_active_clique_index + 1 < best_result {
        best_result = g.max_active_clique_index + 1;
        println!("\nNew best result: {} (vs {})", best_result, num_cliques);
        println!("{}", g.to_string());
      }
      g.conform_cliques_to_vertices();
      g.shuffle_active_cliques();
    }
  }
  //println!("{}", g.to_string());
}

//cargo run --release 205 12 0.75 100_000_000_000 0
