resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-alb"
  description = "HTTPS clients to the Scone load balancer"
  vpc_id      = var.vpc_id
  tags        = local.tags
}

resource "aws_security_group" "api" {
  name        = "${var.name_prefix}-api"
  description = "Scone API reached only through its ALB"
  vpc_id      = var.vpc_id
  tags        = local.tags
}

resource "aws_security_group" "qdrant" {
  name        = "${var.name_prefix}-qdrant"
  description = "Qdrant task private REST access from the API only"
  vpc_id      = var.vpc_id
  tags        = local.tags
}

resource "aws_security_group" "host" {
  name        = "${var.name_prefix}-qdrant-host"
  description = "No inbound host access; no SSH key or public IP"
  vpc_id      = var.vpc_id
  tags        = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "https" {
  for_each          = var.ingress_cidrs
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = each.value
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

resource "aws_vpc_security_group_egress_rule" "alb_api" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.api.id
  ip_protocol                  = "tcp"
  from_port                    = 7437
  to_port                      = 7437
}

resource "aws_vpc_security_group_ingress_rule" "api" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                  = "tcp"
  from_port                    = 7437
  to_port                      = 7437
}

resource "aws_vpc_security_group_egress_rule" "api_qdrant" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.qdrant.id
  ip_protocol                  = "tcp"
  from_port                    = 6333
  to_port                      = 6333
}

resource "aws_vpc_security_group_ingress_rule" "qdrant" {
  security_group_id            = aws_security_group.qdrant.id
  referenced_security_group_id = aws_security_group.api.id
  ip_protocol                  = "tcp"
  from_port                    = 6333
  to_port                      = 6333
}

resource "aws_vpc_security_group_egress_rule" "https" {
  for_each = { for pair in setproduct(["api", "qdrant", "host"], var.https_egress_cidrs) : "${pair[0]}:${pair[1]}" => { group = pair[0], cidr = pair[1] } }
  security_group_id = {
    api    = aws_security_group.api.id
    qdrant = aws_security_group.qdrant.id
    host   = aws_security_group.host.id
  }[each.value.group]
  cidr_ipv4   = each.value.cidr
  ip_protocol = "tcp"
  from_port   = 443
  to_port     = 443
}

resource "aws_vpc_security_group_egress_rule" "backend" {
  for_each          = { for index, rule in var.api_backend_egress : tostring(index) => rule }
  security_group_id = aws_security_group.api.id
  cidr_ipv4         = each.value.cidr
  ip_protocol       = "tcp"
  from_port         = each.value.port
  to_port           = each.value.port
}
