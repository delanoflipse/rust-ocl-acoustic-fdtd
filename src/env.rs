
pub fn var_num<T: std::str::FromStr>(name: &str, def: Option<T>) -> T {
  let value = dotenv::var(name);
  match value{
    Ok(v) => match v.parse::<T>() {
      Ok(v) => v,
      Err(e) => panic!("Environment variable {name} required and no default given!"),
    },
    Err(e) => match def {
      Some(v) => v,
      None => panic!("Environment variable {name} required and no default given!"),
    }
  }
}


pub fn var_bool(name: &str, def: Option<bool>) -> bool {
  let value = dotenv::var(name);
  match value{
    Ok(v) => v.to_lowercase().eq("true"),
    Err(e) => match def {
      Some(v) => v,
      None => panic!("Environment variable {name} required and no default given!"),
    }
  }
}