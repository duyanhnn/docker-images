CREATE DATABASE IF NOT EXISTS `cinnamon_flax`;
CREATE TABLE IF NOT EXISTS `cinnamon_flax`.`task` (
  `id` varchar(256) NOT NULL,
  `request_id` varchar(256) NOT NULL,
  `image` varchar(100) NOT NULL,
  `file_name` varchar(256) NOT NULL,
  `data` json NOT NULL,
  `created_at` datetime(6) NOT NULL,
  PRIMARY KEY (`request_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;